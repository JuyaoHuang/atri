"""Health check endpoint.

Provides a simple health check endpoint for monitoring and load balancers.

Returns ``{"status": "ok"}`` with HTTP 200 when the service is running.

健康检查端点。

为监控和负载均衡器提供简单的健康检查端点。

当服务运行时返回 ``{"status": "ok"}`` 和 HTTP 200。

Reference: docs/Phase5_执行规格.md §US-SRV-003
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    健康检查端点。

    Returns:
        Status dict with "ok" status
        包含 "ok" 状态的状态字典
    """
    return {"status": "ok"}
