from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .agent_schema import ObjectView, View
    from .scene_controller import ControlStepResult, SceneControlModule
except ImportError:
    from agent_schema import ObjectView, View  # type: ignore
    from scene_controller import ControlStepResult, SceneControlModule  # type: ignore


class Motion:
    def __init__(self, scene_controller: SceneControlModule) -> None:
        self.scene_controller = scene_controller

    def _observation_to_view(self, observation: dict[str, Any]) -> View:
        view_id = observation.get("frame_index")
        if not isinstance(view_id, (str, int)):
            view_id = -1
        return View(
            rgb=observation["rgb"],
            depth=observation["depth"],
            camera_to_world=observation["camera_to_world"],
            view_id=view_id,
        )

    def _step_result_to_views(self, step_result: ControlStepResult) -> list[View]:
        observations = step_result.substeps or [step_result.observation]
        return [self._observation_to_view(observation) for observation in observations]

    def _current_view(self) -> View:
        return self._observation_to_view(self.scene_controller.observe())


    def look_around(self) -> list[View]:
        step_result = self.scene_controller.look_right(360.0)
        return self._step_result_to_views(step_result)

    def multiview(self) -> list[View]:
        views: list[View] = []

        self.scene_controller.look_left(60.0)
        self.scene_controller.left(0.25)
        views.append(self._current_view())

        self.scene_controller.look_right(40.0)
        self.scene_controller.right(0.17)
        views.append(self._current_view())

        self.scene_controller.look_right(40.0)
        self.scene_controller.right(0.16)
        views.append(self._current_view())

        self.scene_controller.look_right(40.0)
        self.scene_controller.right(0.17)
        views.append(self._current_view())

        self.scene_controller.look_left(60.0)
        self.scene_controller.left(0.75)

        return views

    def yaw(self) -> list[View]:
        views: list[View] = []

        self.scene_controller.look_left(30.0)
        self.scene_controller.look_up(10.0)
        views.append(self._current_view())

        self.scene_controller.look_down(10.0)
        views.append(self._current_view())

        self.scene_controller.look_down(10.0)
        views.append(self._current_view())

        self.scene_controller.look_right(30.0)
        views.append(self._current_view())

        self.scene_controller.look_up(10.0)
        views.append(self._current_view())

        self.scene_controller.look_up(10.0)
        views.append(self._current_view())

        self.scene_controller.look_right(30.0)
        views.append(self._current_view())

        self.scene_controller.look_down(10.0)
        views.append(self._current_view())

        self.scene_controller.look_down(10.0)
        views.append(self._current_view())

        self.scene_controller.look_left(30.0)
        self.scene_controller.look_up(10.0)

        return views

    def forward(self) -> list[View]:
        step_result = self.scene_controller.forward(1.0)
        return self._step_result_to_views(step_result)

    def backward(self) -> list[View]:
        step_result = self.scene_controller.backward(1.0)
        return self._step_result_to_views(step_result)

    
