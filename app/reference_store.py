class ReferenceStore:
    def __init__(self):
        self._data = {}

    def set_reference(self, model_id, features, feature_types):
        self._data[model_id] = {"features": features, "feature_types": feature_types}

    def get_reference(self, model_id):
        return self._data.get(model_id)

    def has_reference(self, model_id):
        return model_id in self._data