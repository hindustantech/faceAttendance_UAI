# app/services/face_recognition.py (with debug logging)

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from redis import asyncio as aioredis
import json
from bson import ObjectId
import traceback

from app.config import settings
from app.services.face_detection import FaceDetectionService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class FaceRecognitionService:
    """Face Recognition Service with 1:1 and 1:N matching"""

    def __init__(self):
        self.face_detector = FaceDetectionService()
        self.mongo_client = None
        self.redis_client = None
        self.db = None
        self.initialized = False

    async def initialize(self):
        """Initialize connections"""
        try:
            # Initialize MongoDB
            logger.info(f"[DEBUG] Connecting to MongoDB: {settings.MONGODB_URI}")
            logger.info(f"[DEBUG] Database name: {settings.MONGODB_DB_NAME}")

            self.mongo_client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                maxPoolSize=10,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.mongo_client[settings.MONGODB_DB_NAME]

            # Test MongoDB connection
            await self.mongo_client.admin.command('ping')
            logger.info("[DEBUG] MongoDB connection successful")

            # List collections
            collections = await self.db.list_collection_names()
            logger.info(f"[DEBUG] Available collections: {collections}")

            # Try Redis, but continue if not available
            try:
                self.redis_client = await aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis not available: {str(e)}")
                self.redis_client = None

            # Initialize face detector
            await self.face_detector.initialize()

            self.initialized = True
            logger.info("Face recognition service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize recognition service: {str(e)}")
            logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            raise

    async def cleanup(self):
        """Cleanup connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()

    async def verify_face(
        self,
        image_data: np.ndarray,
        company_id: str,
        employee_id: str = None,
        purpose: str = "punch_in",
        device_info: Dict = None
    ) -> Dict:
        """
        Verify a face - supports both 1:1 and 1:N matching
        """
        start_time = datetime.utcnow()

        logger.info(f"[DEBUG] ========== START VERIFICATION ==========")
        logger.info(f"[DEBUG] Company ID: {company_id}")
        logger.info(f"[DEBUG] Employee ID: {employee_id}")
        logger.info(f"[DEBUG] Purpose: {purpose}")
        logger.info(f"[DEBUG] Image shape: {image_data.shape if image_data is not None else 'None'}")

        try:
            # Detect face
            logger.info("[DEBUG] Starting face detection...")
            faces = await self.face_detector.detect_faces(image_data)
            logger.info(f"[DEBUG] Faces detected: {len(faces) if faces else 0}")

            if not faces:
                logger.warning("[DEBUG] No faces detected in image")
                return self._create_error_response(
                    "No face detected in image",
                    start_time
                )

            if len(faces) > 1:
                logger.warning(f"[DEBUG] Multiple faces detected: {len(faces)}")
                return self._create_error_response(
                    "Multiple faces detected. Please provide image with single face.",
                    start_time
                )

            query_face = faces[0]
            logger.info(f"[DEBUG] Face detection score: {query_face.get('det_score')}")
            logger.info(f"[DEBUG] Face embedding exists: {query_face.get('embedding') is not None}")

            if query_face.get('embedding'):
                logger.info(f"[DEBUG] Embedding length: {len(query_face['embedding'])}")
                logger.info(f"[DEBUG] Embedding first 5 values: {query_face['embedding'][:5]}")

            if not query_face.get('embedding'):
                logger.error("[DEBUG] No embedding generated for detected face")
                return self._create_error_response(
                    "Failed to generate face embedding",
                    start_time
                )

            # If employee_id provided, do 1:1 verification
            if employee_id:
                logger.info(f"[DEBUG] Performing 1:1 verification for employee {employee_id}")
                result = await self._verify_specific_employee(
                    query_face,
                    company_id,
                    employee_id,
                    purpose,
                    device_info,
                    start_time
                )
            else:
                # Do 1:N matching against all company employees
                logger.info(f"[DEBUG] Performing 1:N verification for company {company_id}")
                result = await self._verify_against_company(
                    query_face,
                    company_id,
                    purpose,
                    device_info,
                    start_time
                )

            logger.info(f"[DEBUG] Verification result: {result}")
            return result

        except Exception as e:
            logger.error(f"Verification failed: {str(e)}", exc_info=True)
            logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            raise

    async def _verify_specific_employee(
        self,
        query_face: Dict,
        company_id: str,
        employee_id: str,
        purpose: str,
        device_info: Dict,
        start_time: datetime
    ) -> Dict:
        """1:1 verification - verify if face matches specific employee"""
        try:
            logger.info(f"[DEBUG] ========== 1:1 VERIFICATION ==========")
            logger.info(f"[DEBUG] Company ID: {company_id}")
            logger.info(f"[DEBUG] Employee ID: {employee_id}")

            # Get specific employee's enrolled faces
            logger.info("[DEBUG] Fetching employee faces from database...")
            employee_faces = await self._get_employee_faces(company_id, employee_id)

            logger.info(f"[DEBUG] Employee faces found: {employee_faces is not None}")
            if employee_faces:
                logger.info(f"[DEBUG] Employee document keys: {list(employee_faces.keys())}")
                logger.info(f"[DEBUG] isEnrolled: {employee_faces.get('isEnrolled')}")
                logger.info(f"[DEBUG] enrollmentStatus: {employee_faces.get('enrollmentStatus')}")
                logger.info(f"[DEBUG] Number of images: {len(employee_faces.get('images', []))}")
                logger.info(f"[DEBUG] isLocked: {employee_faces.get('isLocked')}")
                logger.info(f"[DEBUG] Employee ID in doc: {employee_faces.get('employeeId')}")
                logger.info(f"[DEBUG] Company ID in doc: {employee_faces.get('companyId')}")

                # Check each image
                for idx, image in enumerate(employee_faces.get('images', [])):
                    logger.info(f"[DEBUG] Image {idx}: isActive={image.get('isActive')}, "
                              f"hasEmbedding={image.get('embedding') is not None}, "
                              f"quality={image.get('quality')}, "
                              f"detScore={image.get('detScore')}")
                    if image.get('embedding'):
                        logger.info(f"[DEBUG] Image {idx} embedding length: {len(image['embedding'])}")
            else:
                logger.warning(f"[DEBUG] No employee document found for employee_id: {employee_id}")

            if not employee_faces:
                logger.error(f"[DEBUG] No enrolled faces found for employee {employee_id}")
                return self._create_error_response(
                    f"No enrolled faces found for employee {employee_id}",
                    start_time
                )

            # Check if employee is locked
            if employee_faces.get('isLocked'):
                logger.warning(f"[DEBUG] Employee {employee_id} account is locked: {employee_faces.get('lockReason')}")
                return self._create_error_response(
                    f"Employee {employee_id} account is locked: {employee_faces.get('lockReason', 'Unknown reason')}",
                    start_time
                )

            # Extract embeddings from employee's images
            enrolled_embeddings = []
            for idx, image in enumerate(employee_faces.get('images', [])):
                logger.info(f"[DEBUG] Processing image {idx}:")
                logger.info(f"[DEBUG]   isActive: {image.get('isActive')}")
                logger.info(f"[DEBUG]   has embedding: {image.get('embedding') is not None}")

                if image.get('isActive') and image.get('embedding'):
                    # FIX: include employee_id so downstream code (logging in
                    # _find_best_match) doesn't KeyError on a missing key.
                    # Previously this dict had no 'employee_id' key at all,
                    # which caused _find_best_match to throw a KeyError on
                    # every candidate face, get silently swallowed by its
                    # try/except, and never update best_match/best_similarity
                    # - even when the true similarity was ~0.99.
                    embedding_data = {
                        'embedding': image['embedding'],
                        'employee_id': employee_id,
                        'image_id': str(image.get('_id', '')),
                        'quality': image.get('quality', 'good')
                    }
                    enrolled_embeddings.append(embedding_data)
                    logger.info(f"[DEBUG]   Added embedding, image_id: {embedding_data['image_id']}")
                else:
                    logger.warning(f"[DEBUG]   Image {idx} skipped: isActive={image.get('isActive')}, "
                                 f"hasEmbedding={image.get('embedding') is not None}")

            logger.info(f"[DEBUG] Total valid enrolled embeddings: {len(enrolled_embeddings)}")

            if not enrolled_embeddings:
                logger.error("[DEBUG] No valid face embeddings found for employee")
                return self._create_error_response(
                    "No valid face embeddings found for employee",
                    start_time
                )

            # Compare with all employee's enrolled faces
            logger.info("[DEBUG] Starting face comparison...")
            best_match = await self._find_best_match(
                query_face['embedding'],
                enrolled_embeddings
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            if best_match:
                logger.info(f"[DEBUG] Best match found:")
                logger.info(f"[DEBUG]   Employee ID: {best_match.get('employee_id')}")
                logger.info(f"[DEBUG]   Similarity: {best_match.get('similarity')}")
                logger.info(f"[DEBUG]   Image ID: {best_match.get('image_id')}")
                logger.info(f"[DEBUG]   Quality: {best_match.get('quality')}")
            else:
                logger.warning("[DEBUG] No match found (best_match is None)")

            # Check if match meets threshold
            is_matched = best_match and best_match['similarity'] >= settings.FACE_MATCH_THRESHOLD
            logger.info(f"[DEBUG] Match threshold: {settings.FACE_MATCH_THRESHOLD}")
            logger.info(f"[DEBUG] Is matched: {is_matched}")

            # Log the verification attempt
            await self._log_verification(
                employee_id,
                company_id,
                is_matched,
                best_match['similarity'] if best_match else 0,
                query_face['det_score'],
                purpose,
                device_info
            )

            # Update consecutive failed attempts
            if not is_matched:
                logger.warning(f"[DEBUG] Verification failed, handling failed attempt")
                await self._handle_failed_attempt(company_id, employee_id)

            result = {
                'success': True,
                'data': {
                    'matched': is_matched,
                    'employee_id': employee_id,
                    'verification_type': '1:1',
                    'similarity': round(best_match['similarity'], 4) if best_match else 0,
                    'confidence': round(best_match['similarity'] * 100, 2) if best_match else 0,
                    'det_score': round(query_face['det_score'], 4),
                    'processing_time': round(processing_time, 3),
                    'threshold_used': settings.FACE_MATCH_THRESHOLD,
                    'best_match_image_id': best_match.get('image_id') if best_match else None
                }
            }

            logger.info(f"[DEBUG] Verification result: {json.dumps(result, default=str)}")
            return result

        except Exception as e:
            logger.error(f"1:1 verification failed: {str(e)}", exc_info=True)
            logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            raise

    async def _verify_against_company(
        self,
        query_face: Dict,
        company_id: str,
        purpose: str,
        device_info: Dict,
        start_time: datetime
    ) -> Dict:
        """1:N verification - find best match in company"""
        try:
            logger.info(f"[DEBUG] ========== 1:N VERIFICATION ==========")
            logger.info(f"[DEBUG] Company ID: {company_id}")

            # Get all enrolled faces for company
            logger.info("[DEBUG] Fetching all enrolled faces for company...")
            enrolled_faces = await self._get_enrolled_faces(company_id)

            logger.info(f"[DEBUG] Total enrolled faces found: {len(enrolled_faces)}")

            if enrolled_faces:
                # Log unique employee IDs
                unique_employees = set(f['employee_id'] for f in enrolled_faces)
                logger.info(f"[DEBUG] Unique employees: {len(unique_employees)}")
                logger.info(f"[DEBUG] Employee IDs: {unique_employees}")

            if not enrolled_faces:
                logger.warning("[DEBUG] No enrolled faces found in company")
                return self._create_error_response(
                    "No enrolled faces found in company",
                    start_time
                )

            # Find best match
            logger.info("[DEBUG] Finding best match...")
            best_match = await self._find_best_match(
                query_face['embedding'],
                enrolled_faces
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            if best_match:
                logger.info(f"[DEBUG] Best match: {best_match}")
            else:
                logger.warning("[DEBUG] No match found")

            is_matched = best_match and best_match['similarity'] >= settings.FACE_MATCH_THRESHOLD
            logger.info(f"[DEBUG] Match threshold: {settings.FACE_MATCH_THRESHOLD}")
            logger.info(f"[DEBUG] Is matched: {is_matched}")

            if is_matched:
                # Log successful verification
                await self._log_verification(
                    best_match['employee_id'],
                    company_id,
                    True,
                    best_match['similarity'],
                    query_face['det_score'],
                    purpose,
                    device_info
                )

                # Get top matches for identification
                top_matches = await self._get_top_matches(
                    query_face['embedding'],
                    enrolled_faces,
                    top_k=5
                )

                return {
                    'success': True,
                    'data': {
                        'matched': True,
                        'employee_id': best_match['employee_id'],
                        'verification_type': '1:N',
                        'similarity': round(best_match['similarity'], 4),
                        'confidence': round(best_match['similarity'] * 100, 2),
                        'det_score': round(query_face['det_score'], 4),
                        'processing_time': round(processing_time, 3),
                        'threshold_used': settings.FACE_MATCH_THRESHOLD,
                        'top_matches': top_matches
                    }
                }
            else:
                # Return top matches even if no match found
                top_matches = await self._get_top_matches(
                    query_face['embedding'],
                    enrolled_faces,
                    top_k=5
                )

                similarity = best_match['similarity'] if best_match else 0
                return {
                    'success': True,
                    'data': {
                        'matched': False,
                        'verification_type': '1:N',
                        'similarity': round(similarity, 4),
                        'processing_time': round(processing_time, 3),
                        'threshold_used': settings.FACE_MATCH_THRESHOLD,
                        'top_matches': top_matches
                    }
                }

        except Exception as e:
            logger.error(f"1:N verification failed: {str(e)}", exc_info=True)
            logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            raise

    async def identify_face(
        self,
        image_data: np.ndarray,
        company_id: str,
        threshold: float = None,
        max_results: int = 10
    ) -> Dict:
        """
        Identify face - find top matches in company
        Always returns top matches regardless of threshold
        """
        start_time = datetime.utcnow()

        try:
            # Detect face
            faces = await self.face_detector.detect_faces(image_data)

            if not faces:
                return self._create_error_response(
                    "No face detected in image",
                    start_time
                )

            query_face = faces[0]

            if not query_face.get('embedding'):
                return self._create_error_response(
                    "Failed to generate face embedding",
                    start_time
                )

            # Get enrolled faces
            enrolled_faces = await self._get_enrolled_faces(company_id)

            if not enrolled_faces:
                return {
                    'success': True,
                    'data': {
                        'matches': [],
                        'total_enrolled': 0,
                        'processing_time': round(
                            (datetime.utcnow() - start_time).total_seconds(), 3
                        )
                    }
                }

            # Use provided threshold or default
            match_threshold = threshold if threshold is not None else settings.FACE_MATCH_THRESHOLD

            # Find all matches
            all_matches = await self._find_all_matches(
                query_face['embedding'],
                enrolled_faces,
                match_threshold
            )

            # Get top matches
            top_matches = all_matches[:max_results]

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                'success': True,
                'data': {
                    'matches': top_matches,
                    'total_matches': len(all_matches),
                    'total_enrolled': len(set(f['employee_id'] for f in enrolled_faces)),
                    'threshold_used': match_threshold,
                    'processing_time': round(processing_time, 3)
                }
            }

        except Exception as e:
            logger.error(f"Identification failed: {str(e)}", exc_info=True)
            raise

    async def _get_employee_faces(self, company_id: str, employee_id: str) -> Optional[Dict]:
        """Get specific employee's enrolled faces"""
        try:
            logger.info(f"[DEBUG] ========== FETCHING EMPLOYEE FACES ==========")
            logger.info(f"[DEBUG] Company ID: {company_id} (type: {type(company_id)})")
            logger.info(f"[DEBUG] Employee ID: {employee_id} (type: {type(employee_id)})")
            logger.info(f"[DEBUG] Database: {self.db.name}")
            logger.info(f"[DEBUG] Collection: faces")

            # Try cache first
            if self.redis_client:
                cache_key = f"employee_faces:{company_id}:{employee_id}"
                logger.info(f"[DEBUG] Checking cache key: {cache_key}")
                cached_data = await self.redis_client.get(cache_key)

                if cached_data:
                    logger.info(f"[DEBUG] Cache HIT for {cache_key}")
                    return json.loads(cached_data)
                else:
                    logger.info(f"[DEBUG] Cache MISS for {cache_key}")

            # Get from MongoDB
            collection = self.db['faces']
            logger.info(f"[DEBUG] MongoDB collection: {collection.name}")
            logger.info(f"[DEBUG] MongoDB database: {collection.database.name}")

            # Try query with string first
            query_string = {
                'companyId': company_id,
                'employeeId': employee_id,
                'isEnrolled': True
            }
            logger.info(f"[DEBUG] Query (string): {json.dumps(query_string, default=str)}")

            document = await collection.find_one(query_string)
            logger.info(f"[DEBUG] Query result (string ID): {'FOUND' if document else 'NOT FOUND'}")

            # If not found, try with ObjectId
            if not document:
                try:
                    query_objectid = {
                        'companyId': ObjectId(company_id) if ObjectId.is_valid(company_id) else company_id,
                        'employeeId': ObjectId(employee_id) if ObjectId.is_valid(employee_id) else employee_id,
                        'isEnrolled': True
                    }
                    logger.info(f"[DEBUG] Query (ObjectId): {json.dumps(query_objectid, default=str)}")

                    document = await collection.find_one(query_objectid)
                    logger.info(f"[DEBUG] Query result (ObjectId): {'FOUND' if document else 'NOT FOUND'}")
                except Exception as e:
                    logger.error(f"[DEBUG] ObjectId query failed: {str(e)}")

            # If still not found, try without isEnrolled filter
            if not document:
                query_no_filter = {
                    'companyId': company_id,
                    'employeeId': employee_id
                }
                logger.info(f"[DEBUG] Query (no isEnrolled filter): {json.dumps(query_no_filter, default=str)}")

                document = await collection.find_one(query_no_filter)
                logger.info(f"[DEBUG] Query result (no filter): {'FOUND' if document else 'NOT FOUND'}")

                if document:
                    logger.info(f"[DEBUG] Document found but isEnrolled might be false: {document.get('isEnrolled')}")
                    logger.info(f"[DEBUG] Document enrollmentStatus: {document.get('enrollmentStatus')}")

            # If still not found, list all documents to debug
            if not document:
                logger.warning(f"[DEBUG] No document found. Listing all documents in collection...")
                all_docs = await collection.find({}).to_list(length=5)
                logger.info(f"[DEBUG] Total documents in collection: {await collection.count_documents({})}")

                for idx, doc in enumerate(all_docs):
                    logger.info(f"[DEBUG] Document {idx}:")
                    logger.info(f"[DEBUG]   companyId: {doc.get('companyId')}")
                    logger.info(f"[DEBUG]   employeeId: {doc.get('employeeId')}")
                    logger.info(f"[DEBUG]   isEnrolled: {doc.get('isEnrolled')}")
                    logger.info(f"[DEBUG]   enrollmentStatus: {doc.get('enrollmentStatus')}")

                return None

            # Convert ObjectId to string for JSON serialization
            logger.info("[DEBUG] Serializing document...")
            result = self._serialize_document(document)
            logger.info(f"[DEBUG] Serialized result keys: {list(result.keys())}")
            logger.info(f"[DEBUG] Employee ID in result: {result.get('employeeId')}")
            logger.info(f"[DEBUG] Is enrolled: {result.get('isEnrolled')}")
            logger.info(f"[DEBUG] Number of images: {len(result.get('images', []))}")

            # Cache if Redis is available
            if self.redis_client and result:
                cache_key = f"employee_faces:{company_id}:{employee_id}"
                logger.info(f"[DEBUG] Caching result with key: {cache_key}")
                await self.redis_client.setex(
                    cache_key,
                    settings.FACE_DATABASE_CACHE_TTL,
                    json.dumps(result)
                )

            logger.info(f"[DEBUG] Successfully retrieved employee faces")
            return result

        except Exception as e:
            logger.error(f"Failed to get employee faces: {str(e)}")
            logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            return None

    async def _get_enrolled_faces(self, company_id: str) -> List[Dict]:
        """Get all enrolled faces for company"""
        try:
            logger.info(f"[DEBUG] ========== FETCHING ALL ENROLLED FACES ==========")
            logger.info(f"[DEBUG] Company ID: {company_id}")

            # Try cache first
            if self.redis_client:
                cache_key = f"enrolled_faces:{company_id}"
                logger.info(f"[DEBUG] Checking cache: {cache_key}")
                cached_data = await self.redis_client.get(cache_key)

                if cached_data:
                    logger.info(f"[DEBUG] Cache HIT")
                    return json.loads(cached_data)
                else:
                    logger.info(f"[DEBUG] Cache MISS")

            # Get from MongoDB
            collection = self.db['faces']
            logger.info(f"[DEBUG] Collection: {collection.name}")

            # Try with string
            query = {
                'companyId': company_id,
                'isEnrolled': True,
                'enrollmentStatus': 'approved',
                'isLocked': {'$ne': True}
            }
            logger.info(f"[DEBUG] Query: {json.dumps(query, default=str)}")

            # First, count total documents
            total_docs = await collection.count_documents({})
            logger.info(f"[DEBUG] Total documents in collection: {total_docs}")

            # Count documents for this company
            company_docs = await collection.count_documents({'companyId': company_id})
            logger.info(f"[DEBUG] Documents for company {company_id}: {company_docs}")

            # Count enrolled documents
            enrolled_count = await collection.count_documents({
                'companyId': company_id,
                'isEnrolled': True
            })
            logger.info(f"[DEBUG] Enrolled documents: {enrolled_count}")

            cursor = collection.find(query)
            logger.info(f"[DEBUG] Cursor created")

            enrolled_faces = []
            doc_count = 0

            async for document in cursor:
                doc_count += 1
                logger.info(f"[DEBUG] Processing document {doc_count}")
                logger.info(f"[DEBUG]   employeeId: {document.get('employeeId')}")
                logger.info(f"[DEBUG]   isEnrolled: {document.get('isEnrolled')}")
                logger.info(f"[DEBUG]   enrollmentStatus: {document.get('enrollmentStatus')}")
                logger.info(f"[DEBUG]   Number of images: {len(document.get('images', []))}")

                employee_id = str(document['employeeId'])

                for idx, image in enumerate(document.get('images', [])):
                    logger.info(f"[DEBUG]   Image {idx}:")
                    logger.info(f"[DEBUG]     isActive: {image.get('isActive')}")
                    logger.info(f"[DEBUG]     hasEmbedding: {image.get('embedding') is not None}")
                    logger.info(f"[DEBUG]     quality: {image.get('quality')}")

                    if image.get('isActive') and image.get('embedding'):
                        enrolled_faces.append({
                            'employee_id': employee_id,
                            'embedding': image['embedding'],
                            'image_id': str(image.get('_id', '')),
                            'quality': image.get('quality', 'good'),
                            'det_score': image.get('detScore', 0)
                        })
                        logger.info(f"[DEBUG]     Added to enrolled faces")

            logger.info(f"[DEBUG] Total documents processed: {doc_count}")
            logger.info(f"[DEBUG] Total enrolled face embeddings: {len(enrolled_faces)}")

            # If no faces found, try without filters to debug
            if not enrolled_faces:
                logger.warning("[DEBUG] No enrolled faces found. Checking all documents...")
                all_docs = await collection.find({'companyId': company_id}).to_list(length=10)
                logger.info(f"[DEBUG] All documents for company: {len(all_docs)}")

                for idx, doc in enumerate(all_docs):
                    logger.info(f"[DEBUG] Doc {idx}: employeeId={doc.get('employeeId')}, "
                              f"isEnrolled={doc.get('isEnrolled')}, "
                              f"enrollmentStatus={doc.get('enrollmentStatus')}, "
                              f"isLocked={doc.get('isLocked')}")

            # Cache if Redis is available
            if self.redis_client and enrolled_faces:
                cache_key = f"enrolled_faces:{company_id}"
                logger.info(f"[DEBUG] Caching {len(enrolled_faces)} faces")
                await self.redis_client.setex(
                    cache_key,
                    settings.FACE_DATABASE_CACHE_TTL,
                    json.dumps(enrolled_faces)
                )

            logger.info(f"[DEBUG] Returning {len(enrolled_faces)} enrolled faces")
            return enrolled_faces

        except Exception as e:
            logger.error(f"Failed to get enrolled faces: {str(e)}")
            logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            return []

    async def _find_best_match(
        self,
        query_embedding: List[float],
        enrolled_faces: List[Dict]
    ) -> Optional[Dict]:
        """Find best matching face using cosine similarity"""
        try:
            logger.info(f"[DEBUG] ========== FINDING BEST MATCH ==========")
            logger.info(f"[DEBUG] Query embedding length: {len(query_embedding) if query_embedding else 0}")
            logger.info(f"[DEBUG] Enrolled faces count: {len(enrolled_faces)}")

            if not query_embedding or not enrolled_faces:
                logger.warning("[DEBUG] No query embedding or enrolled faces")
                return None

            query_vec = np.array(query_embedding, dtype=np.float64)
            # Normalize query vector
            query_vec_norm = np.linalg.norm(query_vec)
            logger.info(f"[DEBUG] Query vector norm before normalization: {query_vec_norm}")
            query_vec = query_vec / query_vec_norm
            logger.info(f"[DEBUG] Query vector norm after normalization: {np.linalg.norm(query_vec)}")

            best_match = None
            best_similarity = -1
            similarities = []

            for idx, face in enumerate(enrolled_faces):
                try:
                    enrolled_vec = np.array(face['embedding'], dtype=np.float64)
                    # Normalize enrolled vector
                    enrolled_vec = enrolled_vec / np.linalg.norm(enrolled_vec)

                    # Calculate cosine similarity
                    similarity = np.dot(query_vec, enrolled_vec)
                    similarities.append(float(similarity))

                    # FIX: use .get() instead of face['employee_id'] so a
                    # missing key never throws mid-loop. A KeyError here used
                    # to be swallowed by the except block below, which meant
                    # best_match/best_similarity never got updated for ANY
                    # face - even when similarity was ~0.99.
                    logger.info(f"[DEBUG] Face {idx}: employee={face.get('employee_id', 'n/a')}, "
                              f"similarity={float(similarity):.4f}, "
                              f"image_id={face.get('image_id', 'n/a')}")

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'employee_id': face.get('employee_id'),
                            'similarity': float(similarity),
                            'image_id': face.get('image_id', ''),
                            'quality': face.get('quality', 'good')
                        }
                        logger.info(f"[DEBUG] New best match: {best_match}")

                except Exception as e:
                    logger.warning(f"Error comparing face {idx}: {str(e)}")
                    logger.warning(f"[DEBUG] Face data: employee_id={face.get('employee_id')}, "
                                 f"embedding_length={len(face.get('embedding', []))}")
                    continue

            if similarities:
                logger.info(f"[DEBUG] All similarities: {[f'{s:.4f}' for s in similarities]}")
                logger.info(f"[DEBUG] Best similarity: {best_similarity:.4f}")
                logger.info(f"[DEBUG] Average similarity: {np.mean(similarities):.4f}")
                logger.info(f"[DEBUG] Similarity range: {min(similarities):.4f} - {max(similarities):.4f}")

            return best_match

        except Exception as e:
            logger.error(f"Face matching error: {str(e)}")
            logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            return None

    async def _find_all_matches(
        self,
        query_embedding: List[float],
        enrolled_faces: List[Dict],
        threshold: float
    ) -> List[Dict]:
        """Find all matching faces above threshold"""
        try:
            if not query_embedding or not enrolled_faces:
                return []

            query_vec = np.array(query_embedding, dtype=np.float64)
            query_vec = query_vec / np.linalg.norm(query_vec)

            # Group similarities by employee
            employee_matches = {}

            for face in enrolled_faces:
                try:
                    enrolled_vec = np.array(face['embedding'], dtype=np.float64)
                    enrolled_vec = enrolled_vec / np.linalg.norm(enrolled_vec)

                    similarity = float(np.dot(query_vec, enrolled_vec))

                    if similarity >= threshold:
                        employee_id = face['employee_id']

                        if employee_id not in employee_matches:
                            employee_matches[employee_id] = {
                                'employee_id': employee_id,
                                'similarity': similarity,
                                'best_similarity': similarity,
                                'num_images_matched': 0,
                                'matched_images': []
                            }

                        # Update best similarity
                        if similarity > employee_matches[employee_id]['best_similarity']:
                            employee_matches[employee_id]['best_similarity'] = similarity
                            employee_matches[employee_id]['similarity'] = similarity

                        employee_matches[employee_id]['num_images_matched'] += 1
                        employee_matches[employee_id]['matched_images'].append({
                            'image_id': face['image_id'],
                            'similarity': similarity,
                            'quality': face.get('quality', 'good')
                        })

                except Exception as e:
                    logger.warning(f"Error comparing face: {str(e)}")
                    continue

            # Convert to list and sort by similarity
            matches_list = list(employee_matches.values())
            matches_list.sort(key=lambda x: x['similarity'], reverse=True)

            # Round similarities
            for match in matches_list:
                match['similarity'] = round(match['similarity'], 4)
                match['best_similarity'] = round(match['best_similarity'], 4)
                for img in match['matched_images']:
                    img['similarity'] = round(img['similarity'], 4)

            return matches_list

        except Exception as e:
            logger.error(f"Find all matches error: {str(e)}")
            return []

    async def _get_top_matches(
        self,
        query_embedding: List[float],
        enrolled_faces: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """Get top K matches"""
        all_matches = await self._find_all_matches(
            query_embedding,
            enrolled_faces,
            threshold=0.0  # Get all matches
        )

        return all_matches[:top_k]

    async def _handle_failed_attempt(self, company_id: str, employee_id: str):
        """Handle failed verification attempt - may lock account"""
        try:
            collection = self.db['faces']

            # Get current failed attempts
            document = await collection.find_one({
                'companyId': company_id,
                'employeeId': employee_id
            })

            if document:
                consecutive_fails = document.get('consecutiveFailedAttempts', 0) + 1

                # FIX: settings.MAX_FAILED_ATTEMPTS did not exist, which
                # raised AttributeError every time this ran (caught below,
                # so it silently never locked accounts). Use getattr with a
                # sane default so this degrades gracefully, but you should
                # also add MAX_FAILED_ATTEMPTS to your Settings/config class,
                # e.g.:
                #     MAX_FAILED_ATTEMPTS: int = 5
                max_failed_attempts = getattr(settings, "MAX_FAILED_ATTEMPTS", 5)

                # Lock account after too many failed attempts
                if consecutive_fails >= max_failed_attempts:
                    await collection.update_one(
                        {'companyId': company_id, 'employeeId': employee_id},
                        {
                            '$set': {
                                'isLocked': True,
                                'lockedAt': datetime.utcnow(),
                                'lockReason': f'Too many failed attempts ({consecutive_fails})'
                            }
                        }
                    )
                    logger.warning(f"Account locked for employee {employee_id}")

        except Exception as e:
            logger.error(f"Failed to handle failed attempt: {str(e)}")

    async def _log_verification(
        self,
        employee_id: str,
        company_id: str,
        matched: bool,
        similarity: float,
        det_score: float,
        purpose: str,
        device_info: Dict = None
    ):
        """Log verification attempt"""
        try:
            collection = self.db['faces']

            log_entry = {
                'attemptedAt': datetime.utcnow(),
                'matched': matched,
                'similarity': similarity,
                'detScore': det_score,
                'purpose': purpose,
                'deviceInfo': device_info or {}
            }

            await collection.update_one(
                {'employeeId': employee_id, 'companyId': company_id},
                {
                    '$push': {
                        'verificationLogs': {
                            '$each': [log_entry],
                            '$slice': -100
                        }
                    },
                    '$set': {'lastVerifiedAt': datetime.utcnow()},
                    '$inc': {
                        'totalVerificationAttempts': 1,
                        'totalFailedAttempts': 0 if matched else 1,
                        'consecutiveFailedAttempts': 0 if matched else 1
                    }
                },
                upsert=True
            )

        except Exception as e:
            logger.error(f"Failed to log verification: {str(e)}")

    def _serialize_document(self, document: Dict) -> Dict:
        """Convert MongoDB document to JSON-serializable format"""
        if document is None:
            return None

        result = {}
        for key, value in document.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = self._serialize_document(value)
            elif isinstance(value, list):
                result[key] = [
                    self._serialize_document(item) if isinstance(item, dict)
                    else str(item) if isinstance(item, ObjectId)
                    else item.isoformat() if isinstance(item, datetime)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    async def invalidate_cache(self, company_id: str = None, employee_id: str = None):
        """Invalidate cache"""
        if self.redis_client:
            try:
                if company_id and employee_id:
                    await self.redis_client.delete(f"employee_faces:{company_id}:{employee_id}")
                elif company_id:
                    await self.redis_client.delete(f"enrolled_faces:{company_id}")
                    # Also delete all employee-specific caches for this company
                    keys = await self.redis_client.keys(f"employee_faces:{company_id}:*")
                    if keys:
                        await self.redis_client.delete(*keys)
                else:
                    keys = await self.redis_client.keys("enrolled_faces:*")
                    keys += await self.redis_client.keys("employee_faces:*")
                    if keys:
                        await self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Failed to invalidate cache: {str(e)}")

    def _create_error_response(self, error: str, start_time: datetime) -> Dict:
        """Create standardized error response"""
        process_time = (datetime.utcnow() - start_time).total_seconds()

        return {
            'success': True,
            'data': {
                'matched': False,
                'error': error,
                'processing_time': round(process_time, 3)
            }
        }