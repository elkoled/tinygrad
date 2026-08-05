import ctypes, unittest
from types import SimpleNamespace
from unittest.mock import patch

from tinygrad.runtime.autogen import libusb
from tinygrad.runtime.ops_amd import AMDAllocator
from tinygrad.runtime.support.usb import alloc_cbuffer, CustomASM24Controller, USB3, USBError, USBIntegrityError, USBMMIOInterface


class FakeUSB:
  def __init__(self, failures=0):
    self.failures, self.events, self.recovery_generation = failures, [], 0

  def reopen(self, delay=0): self.events.append(("reopen", delay))
  def retry(self, fxn, recover=lambda: None): return USB3.retry(self, fxn, recover)

  def control_write(self, request, value=0, index=0): self.events.append(("control", request, value, index))
  def control_read(self, request, length, value=0):
    self.events.append(("read", request, length, value))
    if self.failures:
      self.failures -= 1
      raise USBError("injected", libusb.LIBUSB_ERROR_IO)
    return bytes(length)
  def bulk_write(self, data):
    self.events.append(("bulk", bytes(data)))
    if self.failures:
      self.failures -= 1
      raise USBError("injected", libusb.LIBUSB_ERROR_IO)


class TestUSBRecovery(unittest.TestCase):
  @staticmethod
  def controller(usb):
    controller = CustomASM24Controller.__new__(CustomASM24Controller)
    controller.usb = usb
    return controller

  def test_f2_write_replays_setup_and_data(self):
    usb = FakeUSB(1)
    self.controller(usb).scsi_write(bytes(512))
    transaction = [("control", 0xF2, 1, 1 << 8), ("bulk", bytes(512))]
    self.assertEqual(usb.events, transaction + [("reopen", 0)] + transaction)

  def test_e4_read_retries(self):
    usb = FakeUSB(9)
    self.assertEqual(self.controller(usb).read(0x1234, 8), bytes(8))
    self.assertEqual(sum(event[0] == "reopen" for event in usb.events), 9)
    self.assertEqual(usb.recovery_generation, 9)

  def test_non_transport_error_is_not_retried(self):
    usb = FakeUSB()
    with self.assertRaisesRegex(USBError, "removed"):
      usb.retry(lambda: (_ for _ in ()).throw(USBError("removed", libusb.LIBUSB_ERROR_NO_DEVICE)))
    self.assertFalse(any(event[0] == "reopen" for event in usb.events))

  def test_retry_limit(self):
    usb = FakeUSB(10)
    with self.assertRaisesRegex(USBError, "injected"): self.controller(usb).read(0, 1)
    self.assertEqual(sum(event[0] == "reopen" for event in usb.events), 9)

  def test_short_bulk_transfers_are_retryable_errors(self):
    usb = USB3.__new__(USB3)
    usb.handle, usb._transferred = None, ctypes.c_int(0)
    usb._bulk_buf, usb._bulk_mv = alloc_cbuffer(8)
    def short_transfer(_handle, _endpoint, _buf, length, transferred, _timeout):
      transferred.value = length - 1
      return 0
    with patch.object(libusb, "libusb_bulk_transfer", short_transfer):
      with self.assertRaises(USBError) as out_error: usb.bulk_write(b"1234")
      with self.assertRaises(USBError) as in_error: usb.bulk_read(4)
    self.assertEqual((out_error.exception.rc, in_error.exception.rc), (libusb.LIBUSB_ERROR_IO, libusb.LIBUSB_ERROR_IO))


class FakeController:
  def __init__(self, failures): self.usb, self.failures, self.writes = FakeUSB(), failures, []
  def pcie_mem_write(self, address, data):
    self.writes.append((address, bytes(data)))
    if self.failures:
      self.failures -= 1
      raise USBError("injected", libusb.LIBUSB_ERROR_IO)


class TestMemoryWriteRecovery(unittest.TestCase):
  def test_replay_is_explicit(self):
    controller = FakeController(2)
    USBMMIOInterface(controller, 0x1000, 4, 'I', replay_writes=True)[0] = 42
    self.assertEqual(len(controller.writes), 3)

    controller = FakeController(1)
    with self.assertRaisesRegex(USBError, "injected"): USBMMIOInterface(controller, 0x1000, 4, 'I')[0] = 42
    self.assertEqual(len(controller.writes), 1)


class FakeIntegrityAllocator(AMDAllocator):
  def __init__(self, uploads=(), downloads=()):
    self.usb = SimpleNamespace(recovery_generation=0)
    self.dev = SimpleNamespace(is_usb=lambda: True,
      iface=SimpleNamespace(pci_dev=SimpleNamespace(usb=SimpleNamespace(usb=self.usb))))
    self.uploads, self.downloads = list(uploads), list(downloads)
    self.copyin_count = self.copyout_count = 0
    self.data = b''

  def _copyin_once(self, dest, src):
    self.copyin_count += 1
    action = self.uploads.pop(0) if self.uploads else "clean"
    if action.startswith("recover"): self.usb.recovery_generation += 1
    self.data = bytes(src)
    if action.endswith("corrupt"): self.data = bytes([self.data[0] ^ 1]) + self.data[1:]

  def _copyout_once(self, dest, src):
    self.copyout_count += 1
    recover, data = self.downloads.pop(0) if self.downloads else (False, self.data)
    if recover: self.usb.recovery_generation += 1
    dest.cast('B')[:] = data


class TestUSBTransferIntegrity(unittest.TestCase):
  def test_copyin_without_recovery_has_no_verification(self):
    allocator = FakeIntegrityAllocator()
    allocator._copyin(None, memoryview(b"good"))
    self.assertEqual((allocator.copyin_count, allocator.copyout_count), (1, 0))

  def test_recovered_copyin_is_verified(self):
    allocator = FakeIntegrityAllocator(uploads=["recover"])
    allocator._copyin(None, memoryview(b"good"))
    self.assertEqual((allocator.copyin_count, allocator.copyout_count), (1, 1))

  def test_copyin_rewrite_remains_verified(self):
    allocator = FakeIntegrityAllocator(uploads=["recover_corrupt", "clean"])
    allocator._copyin(None, memoryview(b"good"))
    self.assertEqual((allocator.copyin_count, allocator.copyout_count), (2, 2))

  def test_copyin_verification_with_recovery_is_discarded(self):
    allocator = FakeIntegrityAllocator(uploads=["recover", "clean"], downloads=[(True, b"good"), (False, b"good")])
    allocator._copyin(None, memoryview(b"good"))
    self.assertEqual((allocator.copyin_count, allocator.copyout_count), (2, 2))

  def test_copyin_persistent_corruption_fails_closed(self):
    allocator = FakeIntegrityAllocator(uploads=["recover_corrupt", "corrupt", "corrupt"])
    with self.assertRaises(USBIntegrityError): allocator._copyin(None, memoryview(b"good"))
    self.assertEqual((allocator.copyin_count, allocator.copyout_count), (3, 3))

  def test_recovered_copyout_requires_two_matching_clean_reads(self):
    allocator = FakeIntegrityAllocator(downloads=[(True, b"bad!"), (False, b"good"), (False, b"good")])
    dest = memoryview(bytearray(4))
    allocator._copyout(dest, None)
    self.assertEqual(bytes(dest), b"good")
    self.assertEqual(allocator.copyout_count, 3)

  def test_recovered_copyout_disagreement_fails_closed(self):
    allocator = FakeIntegrityAllocator(downloads=[(True, b"bad!"), (False, b"one!"), (False, b"two!"), (False, b"tri!")])
    with self.assertRaises(USBIntegrityError): allocator._copyout(memoryview(bytearray(4)), None)
    self.assertEqual(allocator.copyout_count, 4)


if __name__ == "__main__": unittest.main()
