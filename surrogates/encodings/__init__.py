from surrogates.encodings.graf import GRAFEncoder
from surrogates.encodings.zcp import ZCPEncoder
from surrogates.encodings.str import StrEncoder


class CombinedEncoder:
    def __init__(self, encoders, return_names=False):
        self.encoders = encoders
        self.return_names = return_names

    def __str__(self):
        return "+".join([k for k in self.encoders.keys()])
    
    def encode(self, root):
        res = {} if self.return_names else []
        for enc in self.encoders.values():
            feats = enc.encode(root)
            if self.return_names:
                res.update(feats)
            else:
                res.extend(feats)
        
        return res

def get_encoder(data_loader, args, return_names=True):
    encoders = []
    if args.use_features:
        encoders.append(('GRAF', GRAFEncoder(args, return_names=return_names)))
    if args.use_zcp:
        encoders.append(('ZCP', ZCPEncoder(data_loader, args.num_classes, args, args.device, return_names=return_names)))
    if args.use_str:
        encoders = []
        encoders.append(('STR', StrEncoder()))

    if len(encoders) == 1:
        return encoders[0][1]
    
    return CombinedEncoder({enc[0]: enc[1] for enc in encoders}, return_names=return_names)
