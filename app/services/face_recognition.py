# app/services/face_recognition.py
# Caching (Redis) removed. Enrollment query relaxed so documents whose
# isEnrolled/enrollmentStatus flags aren't set upstream are still found,
# as long as they have an active image with an embedding.
#
# FIX (2026-07-13): _get_enrolled_faces now matches companyId whether it's
# stored as a string OR an ObjectId (via $in), mirroring the fallback that
# _get_employee_faces already did. Previously the 1:N/identify path only
# matched string-typed companyId, so employees whose document had an
# ObjectId-typed companyId were invisible to identify() even though 1:1
# verify_face() found them fine via its two-step fallback query.
#
# FIX (2026-07-13): Both _get_employee_faces and _get_enrolled_faces now
# use $elemMatch on the images array to ensure only documents with at
# least one active image that has an embedding are returned. Previously,
# _get_employee_faces could return a verification-only document (created
# by _log_verification upsert) that had no images/embeddings, causing a
# false "No valid face embeddings found" error even though the actual
# enrollment document existed separately in the collection.

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
import json
from bson import ObjectId
import traceback

from app.config import settings
from app.services.face_detection import FaceDetectionService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class FaceRecognitionService:
    """Face Recognition Service with 1:1 and 1:N matching (no caching)"""

    def __init__(self):
        self.face_detector = FaceDetectionService()
        self.mongo_client = None
        self.db = None
        self.initialized = False

    async def initialize(self):
        """Initialize connections"""
        try:
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

            collections = await self.db.list_collection_names()
            logger.info(f"[DEBUG] Available collections: {collections}")

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

            if employee_faces.get('isLocked'):
                logger.warning(f"[DEBUG] Employee {employee_id} account is locked: {employee_faces.get('lockReason')}")
                return self._create_error_response(
                    f"Employee {employee_id} account is locked: {employee_faces.get('lockReason', 'Unknown reason')}",
                    start_time
                )

            enrolled_embeddings = []
            for idx, image in enumerate(employee_faces.get('images', [])):
                logger.info(f"[DEBUG] Processing image {idx}:")
                logger.info(f"[DEBUG]   isActive: {image.get('isActive')}")
                logger.info(f"[DEBUG]   has embedding: {image.get('embedding') is not None}")

                if image.get('isActive') and image.get('embedding'):
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

            is_matched = best_match and best_match['similarity'] >= settings.FACE_MATCH_THRESHOLD
            logger.info(f"[DEBUG] Match threshold: {settings.FACE_MATCH_THRESHOLD}")
            logger.info(f"[DEBUG] Is matched: {is_matched}")

            await self._log_verification(
                employee_id,
                company_id,
                is_matched,
                best_match['similarity'] if best_match else 0,
                query_face['det_score'],
                purpose,
                device_info
            )

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

            logger.info("[DEBUG] Fetching all enrolled faces for company...")
            enrolled_faces = await self._get_enrolled_faces(company_id)

            logger.info(f"[DEBUG] Total enrolled faces found: {len(enrolled_faces)}")

            if enrolled_faces:
                unique_employees = set(f['employee_id'] for f in enrolled_faces)
                logger.info(f"[DEBUG] Unique employees: {len(unique_employees)}")
                logger.info(f"[DEBUG] Employee IDs: {unique_employees}")

            if not enrolled_faces:
                logger.warning("[DEBUG] No enrolled faces found in company")
                return self._create_error_response(
                    "No enrolled faces found in company",
                    start_time
                )

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
                await self._log_verification(
                    best_match['employee_id'],
                    company_id,
                    True,
                    best_match['similarity'],
                    query_face['det_score'],
                    purpose,
                    device_info
                )

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

            match_threshold = threshold if threshold is not None else settings.FACE_MATCH_THRESHOLD

            all_matches = await self._find_all_matches(
                query_face['embedding'],
                enrolled_faces,
                match_threshold
            )

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
        """
        Get specific employee's enrolled faces (no cache, no enrollment-flag filter).
        
        FIX (2026-07-13): Added $elemMatch on images array to ensure we only
        return documents that have at least one active image with an embedding.
        This prevents matching a verification-only document that was created
        by _log_verification's upsert, which has no images/embeddings.
        """
        try:
            logger.info(f"[DEBUG] ========== FETCHING EMPLOYEE FACES ==========")
            logger.info(f"[DEBUG] Company ID: {company_id} (type: {type(company_id)})")
            logger.info(f"[DEBUG] Employee ID: {employee_id} (type: {type(employee_id)})")
            logger.info(f"[DEBUG] Database: {self.db.name}")
            logger.info(f"[DEBUG] Collection: faces")

            collection = self.db['faces']
            logger.info(f"[DEBUG] MongoDB collection: {collection.name}")
            logger.info(f"[DEBUG] MongoDB database: {collection.database.name}")

            # Build companyId variants for flexible matching
            company_id_variants = [company_id]
            if ObjectId.is_valid(company_id):
                company_id_variants.append(ObjectId(company_id))
                
            employee_id_variants = [employee_id]
            if ObjectId.is_valid(employee_id):
                employee_id_variants.append(ObjectId(employee_id))

            # CRITICAL FIX: Query must ensure document has images array with 
            # at least one active embedding. This prevents matching the 
            # verification-only document that has no images.
            query = {
                'companyId': {'$in': company_id_variants},
                'employeeId': {'$in': employee_id_variants},
                'images': {
                    '$elemMatch': {
                        'isActive': True,
                        'embedding': {'$exists': True, '$ne': None}
                    }
                }
            }
            logger.info(f"[DEBUG] Query (with $elemMatch): {json.dumps(query, default=str)}")

            document = await collection.find_one(query)
            logger.info(f"[DEBUG] Query result: {'FOUND' if document else 'NOT FOUND'}")

            if not document:
                # Try with string IDs only as fallback
                query_string = {
                    'companyId': company_id,
                    'employeeId': employee_id,
                    'images': {
                        '$elemMatch': {
                            'isActive': True,
                            'embedding': {'$exists': True, '$ne': None}
                        }
                    }
                }
                logger.info(f"[DEBUG] Fallback query (string only): {json.dumps(query_string, default=str)}")
                
                document = await collection.find_one(query_string)
                logger.info(f"[DEBUG] Fallback query result: {'FOUND' if document else 'NOT FOUND'}")

            if not document:
                logger.warning(f"[DEBUG] No document with embeddings found. Listing all documents for employee...")
                # Debug: show all documents for this employee (including verification-only ones)
                all_docs = await collection.find({
                    'employeeId': {'$in': employee_id_variants}
                }).to_list(length=10)
                
                logger.info(f"[DEBUG] Total documents for employee: {len(all_docs)}")
                for idx, doc in enumerate(all_docs):
                    num_images = len(doc.get('images', []))
                    has_active_embeddings = any(
                        img.get('isActive') and img.get('embedding') 
                        for img in doc.get('images', [])
                    )
                    logger.info(f"[DEBUG] Document {idx}:")
                    logger.info(f"[DEBUG]   _id: {doc.get('_id')}")
                    logger.info(f"[DEBUG]   companyId: {doc.get('companyId')} (type: {type(doc.get('companyId')).__name__})")
                    logger.info(f"[DEBUG]   isEnrolled: {doc.get('isEnrolled')}")
                    logger.info(f"[DEBUG]   enrollmentStatus: {doc.get('enrollmentStatus')}")
                    logger.info(f"[DEBUG]   numImages: {num_images}")
                    logger.info(f"[DEBUG]   hasActiveEmbeddings: {has_active_embeddings}")
                    logger.info(f"[DEBUG]   keys: {list(doc.keys())}")
                
                return None

            # Log what we found
            logger.info(f"[DEBUG] Found document with embeddings. Serializing...")
            logger.info(f"[DEBUG] Document keys: {list(document.keys())}")
            logger.info(f"[DEBUG] isEnrolled: {document.get('isEnrolled')}")
            logger.info(f"[DEBUG] enrollmentStatus: {document.get('enrollmentStatus')}")
            logger.info(f"[DEBUG] Number of images: {len(document.get('images', []))}")
            
            # Log image details
            for idx, image in enumerate(document.get('images', [])):
                logger.info(f"[DEBUG] Image {idx}: isActive={image.get('isActive')}, "
                          f"hasEmbedding={image.get('embedding') is not None}, "
                          f"quality={image.get('quality')}")
                if image.get('embedding'):
                    logger.info(f"[DEBUG] Image {idx} embedding length: {len(image['embedding'])}")

            logger.info("[DEBUG] Serializing document...")
            result = self._serialize_document(document)
            logger.info(f"[DEBUG] Serialized result keys: {list(result.keys())}")
            logger.info(f"[DEBUG] Employee ID in result: {result.get('employeeId')}")
            logger.info(f"[DEBUG] Is enrolled: {result.get('isEnrolled')}")
            logger.info(f"[DEBUG] Number of images: {len(result.get('images', []))}")

            logger.info(f"[DEBUG] Successfully retrieved employee faces with embeddings")
            return result

        except Exception as e:
            logger.error(f"Failed to get employee faces: {str(e)}")
            logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            return None

    async def _get_enrolled_faces(self, company_id: str) -> List[Dict]:
        """
        Get all enrolled faces for company (no cache, no enrollment-flag filter).

        FIX: match companyId whether it's stored in Mongo as a string OR
        an ObjectId, using $in. Previously this only matched string-typed
        companyId, which meant employees whose document had companyId
        stored as an ObjectId were silently invisible to identify()/1:N,
        even though 1:1 verify_face() found them fine via its separate
        string-then-ObjectId fallback query in _get_employee_faces.
        
        FIX (2026-07-13): Added $elemMatch on images array to ensure we only
        return documents that have at least one active image with an embedding.
        This prevents including verification-only documents in the results.
        """
        try:
            logger.info(f"[DEBUG] ========== FETCHING ALL ENROLLED FACES ==========")
            logger.info(f"[DEBUG] Company ID: {company_id}")

            collection = self.db['faces']
            logger.info(f"[DEBUG] Collection: {collection.name}")

            # Match companyId as either a string or ObjectId in one query.
            company_id_variants = [company_id]
            if ObjectId.is_valid(company_id):
                company_id_variants.append(ObjectId(company_id))

            # CRITICAL FIX: Query must ensure documents have images array with
            # at least one active embedding. This prevents including
            # verification-only documents that have no images.
            query = {
                'companyId': {'$in': company_id_variants},
                'isLocked': {'$ne': True},
                'images': {
                    '$elemMatch': {
                        'isActive': True,
                        'embedding': {'$exists': True, '$ne': None}
                    }
                }
            }
            logger.info(f"[DEBUG] Query: {json.dumps(query, default=str)}")

            total_docs = await collection.count_documents({})
            logger.info(f"[DEBUG] Total documents in collection: {total_docs}")

            company_docs = await collection.count_documents({'companyId': {'$in': company_id_variants}})
            logger.info(f"[DEBUG] Documents for company {company_id}: {company_docs}")

            cursor = collection.find(query)
            logger.info(f"[DEBUG] Cursor created")

            enrolled_faces = []
            doc_count = 0

            async for document in cursor:
                doc_count += 1
                logger.info(f"[DEBUG] Processing document {doc_count}")
                logger.info(f"[DEBUG]   _id: {document.get('_id')}")
                logger.info(f"[DEBUG]   companyId: {document.get('companyId')} (type: {type(document.get('companyId')).__name__})")
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

            if not enrolled_faces:
                logger.warning("[DEBUG] No enrolled faces found. Checking all documents...")
                all_docs = await collection.find({'companyId': {'$in': company_id_variants}}).to_list(length=10)
                logger.info(f"[DEBUG] All documents for company: {len(all_docs)}")

                for idx, doc in enumerate(all_docs):
                    num_images = len(doc.get('images', []))
                    has_active_embeddings = any(
                        img.get('isActive') and img.get('embedding') 
                        for img in doc.get('images', [])
                    )
                    logger.info(f"[DEBUG] Doc {idx}: _id={doc.get('_id')}, "
                              f"companyId={doc.get('companyId')} ({type(doc.get('companyId')).__name__}), "
                              f"employeeId={doc.get('employeeId')}, "
                              f"isEnrolled={doc.get('isEnrolled')}, "
                              f"enrollmentStatus={doc.get('enrollmentStatus')}, "
                              f"isLocked={doc.get('isLocked')}, "
                              f"numImages={num_images}, "
                              f"hasActiveEmbeddings={has_active_embeddings}")

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
        """
        Find best matching face using Euclidean distance, converted to a
        0-1 'similarity' score for compatibility with the rest of the code.

        IMPORTANT: these are 128-d embeddings (consistent with dlib's
        face_recognition library), which are designed to be compared with
        Euclidean distance, NOT cosine similarity. Cosine similarity on
        these vectors compresses all comparisons - same person or not -
        into a narrow high band (roughly 0.85-0.99), which is why
        completely different faces were being reported as matches.

        dlib's own recommended threshold is a Euclidean distance of 0.6
        (lower distance = more similar). We convert distance to a
        'similarity' score via similarity = 1 - distance so the rest of
        the codebase (which expects higher-is-better, threshold ~0.65)
        keeps working, but calibrate FACE_MATCH_THRESHOLD in settings to
        match this new scale - see note in _verify_specific_employee /
        _verify_against_company call sites. A distance of 0.6 corresponds
        to similarity 0.40, so settings.FACE_MATCH_THRESHOLD should be
        updated to something like 0.45-0.50 (stricter) rather than 0.65.
        """
        try:
            logger.info(f"[DEBUG] ========== FINDING BEST MATCH (Euclidean) ==========")
            logger.info(f"[DEBUG] Query embedding length: {len(query_embedding) if query_embedding else 0}")
            logger.info(f"[DEBUG] Enrolled faces count: {len(enrolled_faces)}")

            if not query_embedding or not enrolled_faces:
                logger.warning("[DEBUG] No query embedding or enrolled faces")
                return None

            query_vec = np.array(query_embedding, dtype=np.float64)

            best_match = None
            best_similarity = -1
            distances = []

            for idx, face in enumerate(enrolled_faces):
                try:
                    enrolled_vec = np.array(face['embedding'], dtype=np.float64)

                    # Euclidean (L2) distance - lower means more similar
                    distance = float(np.linalg.norm(query_vec - enrolled_vec))
                    similarity = 1.0 - distance
                    distances.append(distance)

                    logger.info(f"[DEBUG] Face {idx}: employee={face.get('employee_id', 'n/a')}, "
                              f"distance={distance:.4f}, similarity={similarity:.4f}, "
                              f"image_id={face.get('image_id', 'n/a')}")

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'employee_id': face.get('employee_id'),
                            'similarity': float(similarity),
                            'distance': distance,
                            'image_id': face.get('image_id', ''),
                            'quality': face.get('quality', 'good')
                        }
                        logger.info(f"[DEBUG] New best match: {best_match}")

                except Exception as e:
                    logger.warning(f"Error comparing face {idx}: {str(e)}")
                    logger.warning(f"[DEBUG] Face data: employee_id={face.get('employee_id')}, "
                                 f"embedding_length={len(face.get('embedding', []))}")
                    continue

            if distances:
                logger.info(f"[DEBUG] All distances: {[f'{d:.4f}' for d in distances]}")
                logger.info(f"[DEBUG] Best distance: {min(distances):.4f}")
                logger.info(f"[DEBUG] Average distance: {np.mean(distances):.4f}")
                logger.info(f"[DEBUG] Distance range: {min(distances):.4f} - {max(distances):.4f}")

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
        """
        Find all matching faces above threshold, using Euclidean distance
        converted to a similarity score (see _find_best_match docstring
        for why cosine similarity is wrong for these 128-d embeddings).
        """
        try:
            if not query_embedding or not enrolled_faces:
                return []

            query_vec = np.array(query_embedding, dtype=np.float64)

            employee_matches = {}

            for face in enrolled_faces:
                try:
                    enrolled_vec = np.array(face['embedding'], dtype=np.float64)

                    distance = float(np.linalg.norm(query_vec - enrolled_vec))
                    similarity = 1.0 - distance

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

            matches_list = list(employee_matches.values())
            matches_list.sort(key=lambda x: x['similarity'], reverse=True)

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

            document = await collection.find_one({
                'companyId': company_id,
                'employeeId': employee_id
            })

            if document:
                consecutive_fails = document.get('consecutiveFailedAttempts', 0) + 1

                # settings.MAX_FAILED_ATTEMPTS may not exist on older configs;
                # default to 5 if it's missing rather than raising.
                max_failed_attempts = getattr(settings, "MAX_FAILED_ATTEMPTS", 5)

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