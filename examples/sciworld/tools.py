from scienceworld import ScienceWorldEnv
from agent.tools import Tool
from typing import Union, Dict


class ActivateTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Activate"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step("activate " + obj.strip().strip('"').strip("'"))

    def description(self) -> str:
        return "Activate(obj), activate something. For example, Activate(stove) to activate the stove."


class CloseTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Close"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step("close " + obj.strip().strip('"').strip("'"))

    def description(self) -> str:
        return "Close(obj), close something. For example, Close(fridge) to close the fridge."


class DeactivateTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Deactivate"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step("deactivate " + obj.strip().strip('"').strip("'"))

    def description(self) -> str:
        return "Deactivate(obj), deactivate something. For example, Deactivate(lighter) to deactivate the lighter."


class DunkTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Dunk"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj1, obj2 = invoke_data.split(",")
        return self.sciEnv.step(f"dunk {obj1} in {obj2}")

    def description(self) -> str:
        return "Dunk(obj1, obj2), dunk obj1 in obj2. For example, Dunk(soap, bowl) to dunk soup in the bowl."


class EatTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Eat"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"eat {obj}")

    def description(self) -> str:
        return "Eat(obj), eat obj. For example, Eat(potato) to eat the potato."


class FlushTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Flush"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"flush {obj}")

    def description(self) -> str:
        return "Flush(obj), flush obj. For example, Flush(sink) to flush the sink."


class FocusOnTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "FocusOn"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"focus on {obj}")

    def description(self) -> str:
        return "FocusOn(obj), focus on obj. For example, FocusOn(painting) to focus on the painting."


class GoTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Go"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"go {obj}")

    def description(self) -> str:
        return "Go(obj), go to obj. For example, Go(outside) to move to outside."


class InventoryTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Inventory"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return self.sciEnv.step(f"inventory")

    def description(self) -> str:
        return "Inventory(), show your inventories. For example, you can invoke Inventory() to check out your inventories."


class LookAroundTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "LookAround"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return self.sciEnv.step(f"look around")

    def description(self) -> str:
        return "LookAround(), look around and discover the items in the evironment. For example, you can invoke LookAround() to look around the environment discover the items around you."


class LookAtTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "LookAt"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"look at {obj}")

    def description(self) -> str:
        return "LookAt(obj), look at obj. For example, LookAt(painting) to look at the painting."


class LookInTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "LookIn"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"look in {obj}")

    def description(self) -> str:
        return "LookIn(obj), look in obj. For example, LookIn(hallway) to look in the hallway."


class MixTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Mix"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"mix {obj}")

    def description(self) -> str:
        return "Mix(obj), mix obj. For example, Mix(potato) to mix the potato."


class MoveTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Move"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj1, obj2 = invoke_data.split(",")
        return self.sciEnv.step(f"move {obj1} to {obj2}")

    def description(self) -> str:
        return "Move(obj1, obj2), move obj1 to obj2. For example, Move(thermometer, table) to move the thermometer to the table."


class OpenTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Open"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"open {obj}")

    def description(self) -> str:
        return "Open(obj), open obj. For example, Open(door to hallway) to open the door to the hallway."


class PickUpTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "PickUp"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"pick up {obj}")

    def description(self) -> str:
        return "PickUp(obj), pick up obj. For example, PickUp(potato) to pick up the potato."


class PourTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Pour"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj1, obj2 = invoke_data.split(",")
        return self.sciEnv.step(f"pour {obj1} in {obj2}")

    def description(self) -> str:
        return "Pour(obj1, obj2), pour obj1 in pour2. For example, Pour(orange, sink) to pour orange in the sink."


class PutDownTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "PutDown"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"put down {obj}")

    def description(self) -> str:
        return "PutDown(obj), put down obj. For example, PutDown(orange) to put down the orange."


class ReadTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Read"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"read {obj}")

    def description(self) -> str:
        return "Read(obj), read obj. For example, Read(thermometer) to read the thermometer."


class ResetTaskTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "ResetTask"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return self.sciEnv.step(f"reset task")

    def description(self) -> str:
        return "ResetTask(), reset task. For example, you can invoke ResetTask() to reset the whole task and restart."


class TaskTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Task"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return self.sciEnv.step(f"task")

    def description(self) -> str:
        return "Task(), read the task. For example, you can invoke Task() to get the description of task."


class TeleportTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Teleport"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj = invoke_data
        return self.sciEnv.step(f"teleport {obj}")

    def description(self) -> str:
        return "Teleport(obj), teleport to obj. For example, Teleport(hallway) to teleport to hallway."


class UseTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Use"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        obj1, obj2 = invoke_data.split(",")
        return self.sciEnv.step(f"use {obj1} on {obj2}")

    def description(self) -> str:
        return "Use(obj1, obj2), use obj1 on obj2. For example, Use(thermometer, table) to use thermometer on the table."


class WaitTool(Tool):
    def __init__(self, sciEnv: ScienceWorldEnv):
        super().__init__()
        self.invoke_label = "Wait"
        self.sciEnv = sciEnv

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return self.sciEnv.step(f"wait")

    def description(self) -> str:
        return "Wait(), wait. For example, you can invoke Wait() to wait."
