package gpus

gpus: rtxp6000_8x: #GPU & {
	name:        "RTXP6000-8X"
	gpusPerNode: 8
	nodeType:    "rtxp6000-8x"
	cpu:         112
	memory:      "960Gi"
	useIB:       false
}
