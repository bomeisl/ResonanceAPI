from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import requests
from bs4 import BeautifulSoup
import logging
import traceback
import os
import json

# PythonAnywhere proxy configuration
if 'PYTHONANYWHERE_DOMAIN' in os.environ:
    # We're on PythonAnywhere - configure proxy
    proxy_host = 'proxy.server'
    proxy_port = 3128
    PROXIES = {
        'http': f'http://{proxy_host}:{proxy_port}',
        'https': f'http://{proxy_host}:{proxy_port}',
    }
    IS_PYTHONANYWHERE = True
else:
    PROXIES = None
    IS_PYTHONANYWHERE = False

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


def is_url_allowed(url):
    """Check if URL is allowed on PythonAnywhere"""
    if not IS_PYTHONANYWHERE:
        return True

    for domain in PYTHONANYWHERE_WHITELIST:
        if domain in url:
            return True
    return False


@api_view(['GET'])
def health_check(request):
    '''Health check endpoint'''
    return Response({
        'status': 'healthy',
        'mode': 'lightweight',
        'platform': 'PythonAnywhere' if IS_PYTHONANYWHERE else 'Standard',
        'message': 'Physics Course Recommender API is running',
        'allowed_domains': PYTHONANYWHERE_WHITELIST if IS_PYTHONANYWHERE else 'all'
    })


@api_view(['POST'])
def analyze_physics_text(request):
    '''Main analysis endpoint - supports both URL and direct text input'''
    try:
        url = request.data.get('url')
        text = request.data.get('text')  # Allow direct text input

        if not url and not text:
            return Response(
                {
                    'error': 'Either URL or text is required',
                    'hint': 'Send {"url": "..."} or {"text": "..."}'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # If URL provided, try to fetch it
        if url and not text:
            # Check if URL is allowed on PythonAnywhere
            if IS_PYTHONANYWHERE and not is_url_allowed(url):
                return Response(
                    {
                        'error': f'URL domain not allowed on PythonAnywhere free tier',
                        'allowed_domains': PYTHONANYWHERE_WHITELIST,
                        'suggestion': 'Please use Wikipedia URLs or provide text directly using {"text": "your physics content here"}'
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )

            logger.info(f"Fetching content from {url}")

            try:
                # Use proxy if on PythonAnywhere
                response = requests.get(
                    url,
                    timeout=10,
                    proxies=PROXIES if IS_PYTHONANYWHERE else None,
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
                        'suggestion': 'Try using a Wikipedia URL or provide text directly',
                        'allowed_domains': PYTHONANYWHERE_WHITELIST
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            except requests.RequestException as e:
                return Response(
                    {
                        'error': f'Failed to fetch URL: {str(e)}',
                        'suggestion': 'Try providing the text directly using {"text": "..."}'
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )

        # Limit text length
        text = text[:50000] if text else ''

        if len(text) < 100:
            return Response(
                {
                    'error': 'Insufficient text for analysis',
                    'hint': 'Please provide at least 100 characters of physics-related text'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Extract concepts using lightweight extractor
        logger.info("Extracting concepts...")
        extractor = LightweightConceptExtractor()
        concepts = extractor.extract_concepts(text)
        equations = extractor.extract_equations(text)

        # Map to courses
        logger.info("Mapping to courses...")
        mapper = LightweightCourseMapper()
        recommended_courses = mapper.map_concepts_to_courses(concepts)
        learning_path = mapper.generate_learning_path(recommended_courses)

        # Save analysis (optional)
        try:
            from .models import AnalysisRequest
            analysis = AnalysisRequest.objects.create(
                url=url or 'direct_text_input',
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
            'url': url,
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
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def analyze_text_direct(request):
    '''Direct text analysis endpoint - no URL fetching required'''
    try:
        text = request.data.get('text')

        if not text:
            return Response(
                {'error': 'Text is required', 'hint': 'Send {"text": "your physics content"}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Process the text directly
        request.data['url'] = None  # Set URL to None
        return analyze_physics_text(request)

    except Exception as e:
        return Response(
            {'error': f'Failed to analyze text: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


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
    '''Test endpoint using Wikipedia (whitelisted on PythonAnywhere)'''
    try:
        # Wikipedia is whitelisted on PythonAnywhere
        test_url = "https://en.wikipedia.org/wiki/Quantum_mechanics"

        response = requests.get(
            test_url,
            timeout=10,
            proxies=PROXIES if IS_PYTHONANYWHERE else None,
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
