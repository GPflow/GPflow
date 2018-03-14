from .features import InducingPoints, InducingFeature


class Mof(InducingFeature):
    pass


class SharedIndependentMof(Mof):
    """ Only for testing (TODO(VD) remove)"""
    def __init__(self, feat):
        Mof.__init__(self)
        self.feat = feat
    
    def __len__(self):
        return len(self.feat)


class SeparateIndependentMof(Mof):
    def __init__(self, feat_list):
        Mof.__init__(self)
        self.feat_list = feat_list
    
    def __len__(self):
        return len(self.feat_list[0])


class SeparateMixedMof(Mof):
    def __init__(self, feat_list):
        Mof.__init__(self)
        self.feat_list = feat_list

    def __len__(self):
        pass
        # TODO 