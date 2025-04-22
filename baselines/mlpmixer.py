channel_mixer = """
    sequential[
        sequential[
            routing[permute21, computation[linear_x(4)], identity],
            computation[relu]
        ],
        routing[identity, computation[linear_x(0.25)], permute21]
    ]"""
token_mixer = """
    sequential[
        sequential[
            computation[linear256],
            computation[relu]
        ],
        computation[linear512]
    ]"""
mlpmixer_layer = f"""
    branching(2)[
        clone(2),
        sequential[
            computation[norm],
            sequential[
                {channel_mixer},
                {token_mixer},
            ],
        ],
        computation[identity],
        add(2)
    ]"""
mlpmixer_d2 = f"""
    sequential[
        sequential[
            routing[im2col4k4s0p, computation[linear512], identity],
            computation[pos_enc]
        ],
        sequential[
            {mlpmixer_layer},
            {mlpmixer_layer}
        ]
    ]"""
mlpmixer_d4 = f"""
    sequential[
        sequential[
            routing[im2col4k4s0p, computation[linear512], identity],
            computation[pos_enc]
        ],
        sequential[
            sequential[
                {mlpmixer_layer},
                {mlpmixer_layer}
            ],
            sequential[
                {mlpmixer_layer},
                {mlpmixer_layer}
            ]
        ]
    ]"""
mlpmixer_d8 = f"""
    sequential[
        sequential[
            routing[im2col4k4s0p, computation[linear512], identity],
            computation[pos_enc]
        ],
        sequential[
            sequential[
                sequential[
                    {mlpmixer_layer},
                    {mlpmixer_layer}
                ],
                sequential[
                    {mlpmixer_layer},
                    {mlpmixer_layer}
                ]
            ],
            sequential[
                sequential[
                    {mlpmixer_layer},
                    {mlpmixer_layer}
                ],
                sequential[
                    {mlpmixer_layer},
                    {mlpmixer_layer}
                ]
            ]
        ]
    ]"""
