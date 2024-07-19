import pycuda.driver as cuda
import pycuda.autoinit
import os

# Initialize CUDA device
cuda.init()
dev = cuda.Device(0)  # Assuming we are using device 0

# Retrieve PCI device attributes using PyCUDA
pciBus = dev.get_attribute(cuda.device_attribute.PCI_BUS_ID)
pciDevice = dev.get_attribute(cuda.device_attribute.PCI_DEVICE_ID)
pciDomain = dev.get_attribute(cuda.device_attribute.PCI_DOMAIN_ID)

# Define a function to get the NUMA node ID for the PCI device
def topo_get_numNode(pci_bus, pci_dev, pci_domain):
    fname = f"/sys/bus/pci/devices/0000:{pci_bus:02x}:{pci_dev:02x}.{pci_domain:x}/numa_node"
    try:
        with open(fname, "r") as fp:
            numa_node = int(fp.read().strip())
            return numa_node
    except Exception as e:
        print(f"Error: {e}")
        return -1

# Get the NUMA node ID
numa_node_id = topo_get_numNode(pciBus, pciDevice, pciDomain)
print(f"NUMA Node ID: {numa_node_id}")



""" 

#include <cuda.h>
int pciBus, pciDomain, pciDevice;
cuDeviceGetAttribute(&pciBus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
cuDeviceGetAttribute(&pciDevice, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
cuDeviceGetAttribute(&pciDomain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev);




int topo_get_numNode(int pci_bus, int pci_dev, int pci_domain)
{
    char fname[1024];
    char buff[100];
    int ret = snprintf(fname, 1023, "/sys/bus/pci/devices/0000:%02x:%02x.%1x/numa_node", pci_bus, pci_dev, pci_domain);
    if (ret > 0)
    {
        fname[ret] = '\0';
        FILE* fp = fopen(fname, "r");
        if (fp)
        {
            ret = fread(buff, sizeof(char), 99, fp);
            int numa_node = atoi(buff);
            fclose(fp);
            return numa_node;
        }
    }
    return -1;
}



cpu_set_t cpuset;
CPU_ZERO(&cpuset);
for x in cpulist:
  CPU_SET(x, &cpuset);
sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset); 

"""
