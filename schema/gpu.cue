package schema

#GPU: {
	name:        string
	image:       string
	gpusPerNode: int
	nodeType:    string
	nodeKey:     string | *"node.coreweave.cloud/type"
	cpu:         int
	memory:    string
	ibDevices: string | *""
	useIB:       bool | *true
	mpiEnv: [string]: string
	podAffinity: bool | *false
}
