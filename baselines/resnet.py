# The ResNet18 architecture, represented in einspace
# the MaxPool operation in the stem is replaced by a convolution


resnet_stem_no_maxpool = """
    sequential[
        sequential[
            sequential[
                routing[im2col3k1s1p, computation[linear64], col2im],
                computation[norm]
            ],
            computation[relu]
        ],
        routing[im2col3k2s1p, computation[linear64], col2im]
    ]"""
resnet_conv7x7_stem_no_maxpool = """
    sequential[
        sequential[
            sequential[
                routing[im2col7k2s3p, computation[linear64], col2im],
                computation[norm]
            ],
            computation[relu]
        ],
        routing[im2col3k2s1p, computation[linear64], col2im]
    ]"""
resnet_block = lambda a, b: f"""
    sequential[
        branching(2)[
            clone(2),
            sequential[
                sequential[
                    sequential[
                        routing[im2col3k1s1p, computation[{a}], col2im],
                        computation[norm]
                    ],
                    computation[relu]
                ],
                sequential[
                    routing[im2col3k1s1p, computation[{b}], col2im],
                    computation[norm]
                ]
            ],
            identity,
            add(2)
        ],
        computation[relu]
    ]"""
resnet_strided_block = lambda a, b: f"""
    sequential[
        branching(2)[
            clone(2),
            sequential[
                sequential[
                    sequential[
                        routing[im2col3k2s1p, computation[{a}], col2im],
                        computation[norm]
                    ],
                    computation[relu]
                ],
                sequential[
                    routing[im2col3k1s1p, computation[{b}], col2im],
                    computation[norm]
                ]
            ],
            sequential[
                routing[im2col1k2s0p, computation[{b}], col2im],
                computation[norm]
            ]
            add(2)
        ],
        computation[relu]
    ]"""
resnet18_no_maxpool = f"""
    sequential[
        sequential[
            {resnet_stem_no_maxpool},
            sequential[
                sequential[
                    {resnet_block('linear64', 'linear64')},
                    {resnet_block('linear64', 'linear64')}
                ],
                sequential[
                    {resnet_strided_block('linear128', 'linear128')},
                    {resnet_block('linear128', 'linear128')}
                ]
            ]
        ],
        sequential[
            sequential[
                {resnet_strided_block('linear256', 'linear256')},
                {resnet_block('linear256', 'linear256')}
            ],
            sequential[
                {resnet_strided_block('linear512', 'linear512')},
                {resnet_block('linear512', 'linear512')}
            ]
        ]
    ]"""
resnet18_conv7x7_no_maxpool = f"""
    sequential[
        sequential[
            {resnet_conv7x7_stem_no_maxpool},
            sequential[
                sequential[
                    {resnet_block('linear64', 'linear64')},
                    {resnet_block('linear64', 'linear64')}
                ],
                sequential[
                    {resnet_strided_block('linear128', 'linear128')},
                    {resnet_block('linear128', 'linear128')}
                ]
            ]
        ],
        sequential[
            sequential[
                {resnet_strided_block('linear256', 'linear256')},
                {resnet_block('linear256', 'linear256')}
            ],
            sequential[
                {resnet_strided_block('linear512', 'linear512')},
                {resnet_block('linear512', 'linear512')}
            ]
        ]
    ]"""
