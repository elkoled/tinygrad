import unittest

from tinygrad.runtime.autogen import libusb
from tinygrad.runtime.support.usb import CustomASM24Controller, USB3, USBError, USBMMIOInterface


class FakeUSB:
  def __init__(self, failures=0):
    self.failures, self.events = failures, []

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

  def test_non_transport_error_is_not_retried(self):
    usb = FakeUSB()
    with self.assertRaisesRegex(USBError, "removed"):
      usb.retry(lambda: (_ for _ in ()).throw(USBError("removed", libusb.LIBUSB_ERROR_NO_DEVICE)))
    self.assertFalse(any(event[0] == "reopen" for event in usb.events))

  def test_retry_limit(self):
    usb = FakeUSB(10)
    with self.assertRaisesRegex(USBError, "injected"): self.controller(usb).read(0, 1)
    self.assertEqual(sum(event[0] == "reopen" for event in usb.events), 9)


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


if __name__ == "__main__": unittest.main()
