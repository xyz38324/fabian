"""
Factory function to create CSI tool instances from tool options
"""

from sensession.lib.config import CSICaptureTool
from sensession.lib.csi_tools.ath9k import Ath9k
from sensession.lib.csi_tools.nexmon import Nexmon
from sensession.lib.csi_tools.csi_tool import CsiTool
from sensession.lib.csi_tools.picoscenes import PicoScenes


def instantiate_tool(tool_type: CSICaptureTool) -> CsiTool:
    """
    Factory method for CsiTool dependent on config

    Args:
        tool_type : Type of tool
    """

    tool: "CsiTool | None" = None
    if tool_type == CSICaptureTool.PICOSCENES:
        tool = PicoScenes()
    elif tool_type == CSICaptureTool.NEXMON:
        tool = Nexmon()
    elif tool_type == CSICaptureTool.ATH9K:
        tool = Ath9k()
    else:
        raise NotImplementedError(
            f"Cannot instantiate tool of type {tool_type}. Implement it!"
        )

    return tool
