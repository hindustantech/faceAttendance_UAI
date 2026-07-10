# app/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import time
from datetime import datetime

from app.config import settings
from app.routes import training, recognition, detection, health
from app.services.face_detection import FaceDetectionService
from app.services.face_training import FaceTrainingService
from app.services.face_recognition import FaceRecognitionService
from app.services.anti_spoofing import AntiSpoofingService
from app.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Global service instances
face_training_service = None
face_recognition_service = None
anti_spoofing_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    global face_training_service, face_recognition_service, anti_spoofing_service
    
    # Startup
    logger.info("=" * 50)
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info("=" * 50)
    
    try:
        # Initialize Anti-Spoofing Service
        anti_spoofing_service = AntiSpoofingService()
        await anti_spoofing_service.initialize()
        logger.info("✓ Anti-spoofing service initialized")
        
        # Initialize Face Training Service
        face_training_service = FaceTrainingService()
        await face_training_service.initialize()
        logger.info("✓ Face training service initialized")

        # Initialize Face Recognition Service
        face_recognition_service = FaceRecognitionService()
        await face_recognition_service.initialize()
        logger.info("✓ Face recognition service initialized")
        
        # Initialize Face Detection Service
        face_detection_service = FaceDetectionService()
        await face_detection_service.initialize()
        logger.info("✓ Face detection service initialized")
        
        logger.info("=" * 50)
        logger.info(f"{settings.APP_NAME} is ready to serve requests")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")
    
    if face_recognition_service:
        await face_recognition_service.cleanup()
    
    if face_training_service:
        await face_training_service.cleanup()
    
    logger.info(f"{settings.APP_NAME} shutdown complete")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Face Recognition and Training Microservice for Attendance System",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS Middleware - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

# Request ID Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID and process time to each request"""
    request_id = f"{int(time.time() * 1000)}-{id(request)}"
    request.state.request_id = request_id
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        
        return response
        
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise

# Include Routers
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health Check"]
)
app.include_router(
    detection.router,
    prefix="/api/v1/face-detection",
    tags=["Face Detection"]
)
 
app.include_router(
    training.router,
    prefix="/api/v1/face-training",
    tags=["Face Training"]
)

app.include_router(
    recognition.router,
    prefix="/api/v1/face-recognition",
    tags=["Face Recognition"]
)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "health": "/health",
            "api_docs": "/api/docs",
            "face_training": "/api/v1/face-training",
            "face_recognition": "/api/v1/face-recognition"
        }
    }

# Global Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.status_code,
                "message": str(exc.detail),
                "type": "http_error"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "internal_error",
                "detail": str(exc) if settings.DEBUG else None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Startup Event
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )