from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import requests
from bs4 import BeautifulSoup
import logging
import traceback
import os

# PythonAnywhere detection
IS_PYTHONANYWHERE = 'PYTHONANYWHERE_DOMAIN' in os.environ

# Import lightweight modules (best for PythonAnywhere)
from .lightweight_extractor import LightweightConceptExtractor
from .lightweight_mapper import LightweightCourseMapper

logger = logging.getLogger(__name__)

# Whitelisted domains on PythonAnywhere free tier
PYTHONANYWHERE_WHITELIST = [
    'wikipedia.org',
    'en.wikipedia.org',
    'wikimedia.org',
    'github.com',
    'raw.githubusercontent.com',
]


@api_view(['GET'])
def health_check(request):
    '''Health check endpoint'''
    return Response({
        'status': 'healthy',
        'mode': 'lightweight',
        'platform': 'PythonAnywhere' if IS_PYTHONANYWHERE else 'Standard',
        'message': 'Physics Course Recommender API is running',
        'endpoints': {
            'analyze': '/api/analyze/',
            'analyze_text': '/api/analyze-text/',
            'courses': '/api/courses/',
            'health': '/api/health/'
        }
    })


@api_view(['POST'])
def analyze_physics_text(request):
    '''Main analysis endpoint - supports both URL and direct text input'''
    try:
        # Get data from request
        url = request.data.get('url')
        text = request.data.get('text')
        source_url = request.data.get('source_url', '')

        logger.info(f"Analyze request - URL: {url}, Text length: {len(text) if text else 0}")

        if not url and not text:
            return Response(
                {
                    'error': 'Either URL or text is required',
                    'hint': 'Send {"url": "..."} or {"text": "..."}'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # If URL provided and no text, fetch the URL
        if url and not text:
            # Check if URL is allowed on PythonAnywhere
            if IS_PYTHONANYWHERE:
                allowed = any(domain in url for domain in PYTHONANYWHERE_WHITELIST)
                if not allowed:
                    return Response(
                        {
                            'error': f'URL domain not allowed on PythonAnywhere free tier',
                            'allowed_domains': PYTHONANYWHERE_WHITELIST,
                            'suggestion': 'Please use Wikipedia URLs or provide text directly'
                        },
                        status=status.HTTP_400_BAD_REQUEST
                    )

            logger.info(f"Fetching content from {url}")

            try:
                response = requests.get(
                    url,
                    timeout=10,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                response.raise_for_status()

                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                for element in soup(['script', 'style', 'meta', 'link']):
                    element.decompose()

                text = soup.get_text(separator=' ', strip=True)

            except requests.exceptions.ProxyError as e:
                return Response(
                    {
                        'error': 'Proxy error - domain may not be whitelisted',
                        'details': str(e),
                        'suggestion': 'Try using a Wikipedia URL or provide text directly'
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            except requests.RequestException as e:
                return Response(
                    {
                        'error': f'Failed to fetch URL: {str(e)}',
                        'suggestion': 'Try providing the text directly'
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )

        # Ensure we have text
        if not text:
            return Response(
                {
                    'error': 'No text to analyze',
                    'hint': 'Provide text directly or a valid URL'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Limit text length
        text = text[:50000]

        if len(text) < 100:
            return Response(
                {
                    'error': 'Insufficient text for analysis',
                    'hint': 'Please provide at least 100 characters of physics-related text',
                    'text_length': len(text)
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Extract concepts using lightweight extractor
        logger.info(f"Extracting concepts from {len(text)} characters...")
        extractor = LightweightConceptExtractor()
        concepts = extractor.extract_concepts(text)
        equations = extractor.extract_equations(text)

        logger.info(f"Found {len(concepts)} concepts and {len(equations)} equations")

        # Map to courses
        logger.info("Mapping to courses...")
        mapper = LightweightCourseMapper()
        recommended_courses = mapper.map_concepts_to_courses(concepts)
        learning_path = mapper.generate_learning_path(recommended_courses)

        logger.info(f"Recommended {len(recommended_courses)} courses")

        # Save analysis (optional)
        try:
            from .models import AnalysisRequest
            analysis = AnalysisRequest.objects.create(
                url=url or source_url or 'direct_text_input',
                extracted_text=text[:5000],
                identified_concepts=[c.get('concept', str(c)) for c in concepts[:20]],
                recommended_courses=[c['course_number'] for c in recommended_courses[:10]]
            )
            analysis_id = analysis.id
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            analysis_id = None

        # Response
        return Response({
            'url': url or source_url,
            'analysis_id': analysis_id,
            'mode': 'lightweight',
            'platform': 'PythonAnywhere' if IS_PYTHONANYWHERE else 'Standard',
            'summary': {
                'total_concepts': len(concepts),
                'total_equations': len(equations),
                'text_length': len(text)
            },
            'top_concepts': concepts[:5],
            'sample_equations': equations[:3],
            'recommended_courses': recommended_courses[:10],
            'learning_path': learning_path[:10]
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())

        return Response({
            'error': 'An unexpected error occurred',
            'details': str(e),
            'type': type(e).__name__
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def analyze_text_direct(request):
    '''Direct text analysis endpoint - FIXED VERSION'''
    try:
        text = request.data.get('text')
        source_url = request.data.get('source_url', 'browser-extension')

        if not text:
            return Response(
                {'error': 'Text is required', 'hint': 'Send {"text": "your physics content"}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        logger.info(f"Direct text analysis - length: {len(text)}, source: {source_url}")

        # Create new data dict with text
        new_data = {
            'text': text,
            'source_url': source_url,
            'url': None
        }

        # Create a new request with the text data
        # Instead of calling the other view, process here directly

        # Limit text length
        text = text[:50000]

        if len(text) < 100:
            return Response(
                {
                    'error': 'Insufficient text for analysis',
                    'hint': 'Please provide at least 100 characters',
                    'text_length': len(text)
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Extract concepts
        logger.info("Extracting concepts...")
        extractor = LightweightConceptExtractor()
        concepts = extractor.extract_concepts(text)
        equations = extractor.extract_equations(text)

        # Map to courses
        logger.info("Mapping to courses...")
        mapper = LightweightCourseMapper()
        recommended_courses = mapper.map_concepts_to_courses(concepts)
        learning_path = mapper.generate_learning_path(recommended_courses)

        # Save analysis
        try:
            from .models import AnalysisRequest
            analysis = AnalysisRequest.objects.create(
                url=source_url,
                extracted_text=text[:5000],
                identified_concepts=[c.get('concept', str(c)) for c in concepts[:20]],
                recommended_courses=[c['course_number'] for c in recommended_courses[:10]]
            )
            analysis_id = analysis.id
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            analysis_id = None

        # Return response
        return Response({
            'url': source_url,
            'analysis_id': analysis_id,
            'mode': 'lightweight',
            'platform': 'PythonAnywhere' if IS_PYTHONANYWHERE else 'Standard',
            'summary': {
                'total_concepts': len(concepts),
                'total_equations': len(equations),
                'text_length': len(text)
            },
            'top_concepts': concepts[:5],
            'sample_equations': equations[:3],
            'recommended_courses': recommended_courses[:10],
            'learning_path': learning_path[:10]
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error in analyze_text_direct: {e}")
        logger.error(traceback.format_exc())

        return Response({
            'error': f'Failed to analyze text: {str(e)}',
            'type': type(e).__name__
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def list_courses(request):
    '''List all available courses'''
    try:
        mapper = LightweightCourseMapper()
        courses = []

        for course_num, info in mapper.courses.items():
            courses.append({
                'course_number': course_num,
                'title': info['title'],
                'description': info.get('description', '')[:200],
                'level': info.get('level', 'Unknown'),
                'url': info.get('url', '')
            })

        return Response({
            'total': len(courses),
            'courses': sorted(courses, key=lambda x: x['course_number'])
        })

    except Exception as e:
        logger.error(f"Error listing courses: {e}")
        return Response(
            {'error': 'Failed to list courses'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def test_wikipedia(request):
    '''Test endpoint using Wikipedia'''
    try:
        test_url = "https://en.wikipedia.org/wiki/Quantum_mechanics"

        response = requests.get(
            test_url,
            timeout=10,
            headers={'User-Agent': 'Mozilla/5.0'}
        )

        if response.status_code == 200:
            return Response({
                'status': 'success',
                'message': 'Wikipedia fetch successful',
                'url': test_url,
                'content_length': len(response.content)
            })
        else:
            return Response({
                'status': 'error',
                'status_code': response.status_code
            })

    except Exception as e:
        return Response({
            'status': 'error',
            'details': str(e)
        })