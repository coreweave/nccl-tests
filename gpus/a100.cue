package gpus

gpus: a100: #GPU & {
	name:        "A100"
	gpusPerNode: 8
	nodeType:    "gd-8xa100-i128"
	cpu:         112
	memory:      "1000Gi"
	ibDevices:   "ibp0:1,ibp1:1,ibp2:1,ibp3:1"
}
