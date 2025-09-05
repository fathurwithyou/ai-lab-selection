class BaseEstimator:
    """Base class for all estimators in the library."""

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)

        nested_params = {}
        for key, value in params.items():
            if "__" in key:
                key_parts = key.split("__", 1)
                obj_name, param_name = key_parts[0], key_parts[1]
                if obj_name not in nested_params:
                    nested_params[obj_name] = {}
                nested_params[obj_name][param_name] = value
            else:
                if key not in valid_params:
                    raise ValueError(f"Invalid parameter {key} for estimator {self}")
                setattr(self, key, value)

        for obj_name, obj_params in nested_params.items():
            if obj_name not in valid_params:
                raise ValueError(f"Invalid parameter {obj_name} for estimator {self}")
            nested_obj = getattr(self, obj_name)
            nested_obj.set_params(**obj_params)

        return self

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        import inspect

        init_signature = inspect.signature(cls.__init__)
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        return sorted([p.name for p in parameters])

    def __repr__(self, N_CHAR_MAX=700):
        """Return string representation of the estimator."""

        N_MAX_ELEMENTS_TO_SHOW = 30

        params = {}
        for key, value in self.get_params(deep=False).items():
            if isinstance(value, str) and len(value) > 30:
                value = value[:27] + "..."
            params[key] = value

        params_str = ", ".join(
            f"{k}={v}" for k, v in list(params.items())[:N_MAX_ELEMENTS_TO_SHOW]
        )

        if len(params) > N_MAX_ELEMENTS_TO_SHOW:
            params_str += ", ..."

        result = f"{self.__class__.__name__}({params_str})"

        if len(result) > N_CHAR_MAX:
            result = result[:N_CHAR_MAX] + "..."

        return result
