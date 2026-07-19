import torch
from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan
from fastmcp.utilities.types import Image
from sam3.config import Settings
from sam3.io import save_mask
from sam3.models import load_models
from sam3.schemas.mcp import MCPRequest
from sam3.tracker import Sam3Tracker
from transform.segment import mcp_to_sam3


@lifespan
async def app_lifespan(_server: FastMCP):
    settings = Settings()
    models = load_models(settings)
    tracker = Sam3Tracker(models)
    try:
        yield {"settings": settings, "tracker": tracker}
    finally:
        del tracker
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


mcp = FastMCP("SAM3 Tracker", lifespan=app_lifespan)


@mcp.tool
def segment_image(req: MCPRequest, ctx: Context) -> Image:
    """图像分割。支持 point/box/prev_mask 提示，返回 mask 文件路径。"""
    tracker: Sam3Tracker = ctx.lifespan_context["tracker"]
    settings: Settings = ctx.lifespan_context["settings"]
    
    sam3_req = mcp_to_sam3(req)
    result_mask = tracker.segment(sam3_req)
    mask_path = save_mask(result_mask, settings.output_dir)
    return Image(mask_path)

def main():
    mcp.run(transport="http", host="0.0.0.0", port=8000)

