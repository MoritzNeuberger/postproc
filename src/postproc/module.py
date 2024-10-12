from __future__ import annotations

import modules as mod


class module:
    def __init__(self, inst):
        self.name = inst["name"]
        self.input = inst["input"]
        self.output = inst["output"]
        self.module_name = inst["module"]
        self.para = inst.get("para", {})
        self.module = self._get_module(self.module_name)

    def _get_module(self, module):
        if module == "group_sensitive_volume":
            return mod.m_group_sensitive_volume
        if module == "window":
            return mod.m_window
        if module == "active_volume":
            return mod.m_active_volume
        if module == "sum_energy":
            return mod.m_sum_energy
        if module == "r90_estimator":
            return mod.m_r90_estimator
        if module == "coincidence_window":
            return mod.m_coincidence_window
        if module == "threshold":
            return mod.m_acceptance_range
        error_message = f"{module} not defined."
        raise NotImplementedError(error_message)

    def run(self, processing_variables):
        self.module(self.para, self.input, self.output, processing_variables)
