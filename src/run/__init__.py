from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan

from sam3.models import load_models
from sam3.schemas import SegmentRequest
from sam3.tracker import Sam3Tracker


@lifespan
async def app_lifespan(_server: FastMCP):
    models = load_models()
    tracker = Sam3Tracker(models)
    try:
        yield {"tracker": tracker}
    finally:
        del tracker
        del models


mcp = FastMCP("SAM3 Tracker", lifespan=app_lifespan)


@mcp.tool
def segment_image(req: SegmentRequest, ctx: Context) -> str:
    """图像分割。支持 point/box/prev_mask 提示，返回 mask 文件路径。"""
    tracker: Sam3Tracker = ctx.lifespan_context["tracker"]
    return tracker.segment(req)


if __name__ == "__main__":
    mcp.run()


def main():
    mcp.run()
