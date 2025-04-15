## Changelog

### nvbandwidth 0.8
Bug Fixes:
 * Device Latency Test Accuracy:
   * Fixed an issue where the device_to_device_latency test was incorrectly
     reporting host-device latency instead of device-to-device latency.
   * Host-device latency reports now correctly reflect C2C or PCIe latency
   depending on the system, while device-to-device latency reports focus on
	NVLINK or equivalent inter-device connections.
 * Adjust buffer size threshold use to select which copy kernel is used, for
	more accurate measurements.
 * Add host name to json output
 * Updated README
