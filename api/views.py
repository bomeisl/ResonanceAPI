from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import requests
from bs4 import BeautifulSoup
import logging
import traceback
import platform

# Detect if we're on macOS
IS_MACOS = platform.system() == 'Darwin'

# Import the appropriate modules based on platform
if IS_MACOS:
    print("macOS detected - using lightweight mode for stability")
    from .lightweight_extractor import LightweightConceptExtractor
    from .lightweight_mapper import LightweightCourseMapper

    USE_ADVANCED = False
else:
    # Try to import advanced modules
    try:
        from .advanced_concept_extractor import AdvancedConceptExtractor
        from .embedding_course_mapper import EmbeddingCourseMapper

        USE_ADVANCED = True
        print("Using advanced BERT-based models")
    except ImportError as e:
        print(f"Cannot import advanced models: {e}")
        print("Falling back to lightweight models")
        from .lightweight_extractor import LightweightConceptExtractor
        from .lightweight_mapper import LightweightCourseMapper

        USE_ADVANCED = False

from .models import AnalysisRequest

logger = logging.getLogger(__name__)


@api_view(['GET'])
def health_check(request):
    '''Health check endpoint'''
    return Response({
        'status': 'healthy',
        'mode': 'lightweight' if IS_MACOS else ('advanced' if USE_ADVANCED else 'lightweight'),
        'platform': platform.system(),
        'message': 'Physics Course Recommender API is running'
    })


@api_view(['POST'])
def analyze_physics_text(request):
    '''Main analysis endpoint'''
    try:
        url = request.data.get('url')

        if not url:
            return Response(
                {'error': 'URL is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Fetch content
        logger.info(f"Fetching content from {url}")

        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0'
            })
            response.raise_for_status()
        except requests.RequestException as e:
            return Response(
                {'error': f'Failed to fetch URL: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()

        text = soup.get_text(separator=' ', strip=True)[:50000]

        if len(text) < 100:
            return Response(
                {'error': 'Insufficient text extracted'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Extract concepts
        logger.info("Extracting concepts...")

        if IS_MACOS or not USE_ADVANCED:
            # Use lightweight extractor
            extractor = LightweightConceptExtractor()
            concepts = extractor.extract_concepts(text)
            equations = extractor.extract_equations(text)
        else:
            # Use advanced extractor
            extractor = AdvancedConceptExtractor()
            concepts, equations = extractor.extract_concepts(text)

        # Map to courses
        logger.info("Mapping to courses...")

        if IS_MACOS or not USE_ADVANCED:
            mapper = LightweightCourseMapper()
            recommended_courses = mapper.map_concepts_to_courses(concepts)
        else:
            mapper = EmbeddingCourseMapper()
            recommended_courses = mapper.map_concepts_to_courses(concepts, equations)

        learning_path = mapper.generate_learning_path(recommended_courses)

        # Save analysis
        try:
            analysis = AnalysisRequest.objects.create(
                url=url,
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
            'mode': 'lightweight' if (IS_MACOS or not USE_ADVANCED) else 'advanced',
            'platform': platform.system(),
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
            'platform': platform.system()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def list_courses(request):
    '''List all courses'''
    try:
        if IS_MACOS or not USE_ADVANCED:
            from .lightweight_mapper import LightweightCourseMapper
            mapper = LightweightCourseMapper()
        else:
            from .course_mapper import EmbeddingCourseMapper
            mapper = EmbeddingCourseMapper()

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