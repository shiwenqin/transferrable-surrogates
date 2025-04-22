# The WideResNet16-4 architecture, represented in einspace
# the MaxPool operation in the stem is replaced by a convolution


wideresnet_stem = """
    sequential[
        sequential[
            routing[im2col3k1s1p, computation[linear16], col2im],
            computation[norm]
        ],
        computation[relu]
    ]"""
wideresnet_shortcut_block = lambda a: f"""
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
                routing[im2col3k1s1p, computation[{a}], col2im]
            ],
            routing[im2col1k1s0p, computation[{a}], col2im],
            add(2)
        ],
        sequential[
            computation[norm],
            computation[relu]
        ]
    ]"""
wideresnet_strided_shortcut_block = lambda a: f"""
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
                routing[im2col3k1s1p, computation[{a}], col2im]
            ],
            routing[im2col1k2s0p, computation[{a}], col2im],
            add(2)
        ],
        sequential[
            computation[norm],
            computation[relu]
        ]
    ]"""
wideresnet_identity_shortcut_block = lambda a: f"""
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
                routing[im2col3k1s1p, computation[{a}], col2im]
            ],
            computation[identity],
            add(2)
        ],
        sequential[
            computation[norm],
            computation[relu]
        ]
    ]"""
wideresnet16_4 = f"""
    sequential[
        {wideresnet_stem},
        sequential[
            sequential[
                sequential[
                    {wideresnet_shortcut_block('linear64')},
                    {wideresnet_identity_shortcut_block('linear64')}
                ],
                sequential[
                    {wideresnet_strided_shortcut_block('linear128')},
                    {wideresnet_identity_shortcut_block('linear128')}
                ]
            ],
            sequential[
                {wideresnet_strided_shortcut_block('linear256')},
                {wideresnet_identity_shortcut_block('linear256')}
            ]
        ]
    ]"""
