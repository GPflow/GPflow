from .features import InducingPoints, InducingFeature

class SeparateInducingFeatures(InducingFeature):
    def __init__(self, feat_list):
        """
        We assume that each feature in feat_list has the same M,
        i.e. len(feat) is equal for all feat in feat_list
        """
        self.feat_list = feat_list


class MultiOutputInducingPoints(InducingPoints):
    """
    Z is the same, but u lives in a different space
    -> we need different Kuu, Kuf
    """
    pass
