# app/routes/health.py
from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

@router.get("")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/detailed")
async def detailed_health():
    """Detailed health check with component status"""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from redis import asyncio as aioredis
        
        health_status = {
            "service": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check MongoDB
        try:
            mongo_client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
            await mongo_client.admin.command('ping')
            health_status["components"]["mongodb"] = {"status": "healthy"}
            mongo_client.close()
        except Exception as e:
            health_status["components"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
        
        # Check Redis
        try:
            redis_client = await aioredis.from_url(settings.REDIS_URL)
            await redis_client.ping()
            health_status["components"]["redis"] = {"status": "healthy"}
            await redis_client.close()
        except Exception as e:
            health_status["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
        
        # Overall status
        all_healthy = all(
            comp["status"] == "healthy"
            for comp in health_status["components"].values()
        )
        
        if not all_healthy:
            health_status["service"] = "degraded"
            raise HTTPException(status_code=503, detail=health_status)
        
        return health_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})