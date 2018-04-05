from .features import InducingPoints, InducingFeature


class Mof(InducingFeature):
    """
    Multi Output Feature Class
    Inducing feature for Multi Output GPs
    """
    pass


class SharedIndependentMof(Mof):
    """
    Only for testing (TODO(VD) remove)
    - Inducing features supports independent GPs
    - Independent GPs share the same inducing features
    """
    def __init__(self, feat):
        Mof.__init__(self)
        self.feat = feat
    
    def __len__(self):
        return len(self.feat)


class SeparateIndependentMof(Mof):
    """
    Only for testing (TODO(VD) remove)
    - Inducing features supports independent GPs
    - Each independent GP has its own inducing features
    """
    def __init__(self, feat_list):
        Mof.__init__(self)
        self.feat_list = feat_list
    
    def __len__(self):
        return len(self.feat_list[0])


class MixedKernelSharedMof(SharedIndependentMof):
    """
    Only for testing (TODO(VD) remove)
    - Inducing features supports mixed GPs
    - Mixed GPs share the same inducing points
    """
    pass

