package gpus

gpus: h100: #GPU & {
	name:        "H100"
	gpusPerNode: 8
	nodeType:    "gd-8xh100ib-i128"
	cpu:         110
	memory:      "960Gi"
	ibDevices:   "ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1"
	mpiEnv: SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING: "1"
}
