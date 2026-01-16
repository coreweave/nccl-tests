package gpus

gpus: a40: #GPU & {
	name:        "A40"
	gpusPerNode: 8
	nodeType:    "A40"
	nodeKey:     "gpu.nvidia.com/model"
	cpu:         90
	memory:      "400Gi"
	ibDevices:   "ibp0:1,ibp1:1,ibp2:1,ibp3:1"
}
