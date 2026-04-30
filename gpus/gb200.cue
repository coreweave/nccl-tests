package gpus

gpus: gb200: #GPU & {
	name:        "GB200"
	gpusPerNode: 4
	nodeType:    "gb200-4x"
	cpu:         140
	memory:      "900Gi"
	useIB:       false
	podAffinity: true
	mpiEnv: {
		NVIDIA_IMEX_CHANNELS:       "0"
		NCCL_NET_GDR_C2C:           "1"
		NCCL_MNNVL_ENABLE:          "1"
		NCCL_CUMEM_ENABLE:          "1"
		NCCL_SHM_DISABLE:           "0"
		UCX_TLS:                    "tcp"
		UCX_NET_DEVICES:            "eth0"
		OMPI_MCA_coll_hcoll_enable: "0"
	}
}
