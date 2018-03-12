from .features import InducingPoints, InducingFeature


class Mof(InducingFeature):
    pass


class SharedIndependentMof(Mof):
    def __init__(self, feat):
        Mof.__init__(self)
        self.feat = feat
    
    def __len__(self):
        return len(self.feat)


class SeparateIndependentMof(Mof):
    def __init__(self, feat_list):
        Mof.__init__(self)
        self.feat_list = feat_list


class SeparateMixedMof(Mof):
    def __init__(self, feat_list):
        Mof.__init__(self)
        self.feat_list = feat_list