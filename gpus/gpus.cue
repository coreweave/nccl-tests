package gpus

import "github.com/coreweave/nccl-tests/schema"

// Default image passed via tag from nccl.nu (which reads workflow YAML and git SHA)
defaultImage: string @tag(defaultImage)

// Alias for schema.#GPU with our default image injected
#GPU: schema.#GPU & {
	image: string | *defaultImage
}

gpus: [string]: #GPU
