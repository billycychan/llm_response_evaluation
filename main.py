import time
import json
import requests
import argparse
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from openai import OpenAI

# ----- Domain Models -----

@dataclass
class TestResult:
    """Data class for test results"""
    times: List[float] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    
    @property
    def avg_time(self) -> float:
        return statistics.mean(self.times) if self.times else 0
    
    @property
    def median_time(self) -> float:
        return statistics.median(self.times) if self.times else 0
    
    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0
    
    @property
    def max_time(self) -> float:
        return max(self.times) if self.times else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "times": self.times,
            "statistics": {
                "avg_time": self.avg_time,
                "median_time": self.median_time,
                "min_time": self.min_time,
                "max_time": self.max_time
            }
        }

# ----- LLM Service Interface -----

class LLMService(ABC):
    """Abstract base class for LLM services"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response for the given prompt"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model being used"""
        pass
    
    def run_test(self, prompt: str, num_runs: int = 5) -> TestResult:
        """Run the test multiple times and collect results"""
        print(f"\nTesting {self.__class__.__name__} with model: {self.get_model_name()}")
        result = TestResult()
        
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}...")
            
            start_time = time.time()
            
            try:
                response_text = self.generate_response(prompt)
                end_time = time.time()
                elapsed = end_time - start_time
                
                result.times.append(elapsed)
                result.responses.append(response_text)
                
                print(f"  Time: {elapsed:.2f} seconds")
                # Print first 50 chars of response
                print(f"  Response preview: {response_text[:50]}...")
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        return result

# ----- Concrete LLM Services -----

class OllamaService(LLMService):
    """Ollama LLM service implementation"""
    
    def __init__(self, model: str = "qwen2.5:14b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate_response(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        return response.json().get("response", "")
    
    def get_model_name(self) -> str:
        return self.model

class ChatGPTService(LLMService):
    """ChatGPT LLM service implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    def generate_response(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return completion.choices[0].message.content
    
    def get_model_name(self) -> str:
        return self.model

# ----- Output Services -----

class TestResultVisualizer:
    """Class responsible for visualizing test results"""
    
    @staticmethod
    def plot_comparison(results: Dict[str, Any], filename: str = 'response_time_comparison.png'):
        """Create a bar chart comparing response times"""
        plt.figure(figsize=(10, 6))
        
        # Data for plotting
        models = []
        avg_times = []
        median_times = []
        
        for service_name, result_data in results.items():
            if service_name != "test_date" and service_name != "prompt":
                models.append(f"{service_name}\n({result_data['model']})")
                avg_times.append(result_data["statistics"]["avg_time"])
                median_times.append(result_data["statistics"]["median_time"])
        
        # Creating the bar chart
        x = range(len(models))
        width = 0.35
        
        plt.bar(x, avg_times, width, label='Average Time', color='skyblue')
        plt.bar([i + width for i in x], median_times, width, label='Median Time', color='lightgreen')
        
        plt.ylabel('Response Time (seconds)')
        plt.title('API Response Time Comparison')
        plt.xticks([i + width/2 for i in x], models)
        plt.legend()
        
        plt.savefig(filename)
        plt.show()
        
        print(f"Plot saved as {filename}")

class TestResultRepository:
    """Class responsible for storing test results"""
    
    @staticmethod
    def save_to_json(results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save test results to a JSON file"""
        if not filename:
            filename = f"response_time_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
        return filename

# ----- Test Runner -----

class LLMTestRunner:
    """Class responsible for coordinating the test execution"""
    
    def __init__(self, visualizer: TestResultVisualizer, repository: TestResultRepository):
        self.services = {}
        self.visualizer = visualizer
        self.repository = repository
    
    def add_service(self, name: str, service: LLMService):
        """Add an LLM service to be tested"""
        self.services[name] = service
    
    def run_tests(self, prompt: str, num_runs: int = 5) -> Dict[str, Any]:
        """Run tests for all services and return results"""
        results = {
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt
        }
        
        for name, service in self.services.items():
            test_result = service.run_test(prompt, num_runs)
            results[name] = {
                "model": service.get_model_name(),
                **test_result.to_dict()
            }
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the test results"""
        print("\n----- SUMMARY -----")
        
        for service_name, result_data in results.items():
            if service_name != "test_date" and service_name != "prompt":
                print(f"{service_name} ({result_data['model']}):")
                print(f"  Average time: {result_data['statistics']['avg_time']:.2f} seconds")
                print(f"  Median time: {result_data['statistics']['median_time']:.2f} seconds")
                print(f"  Min time: {result_data['statistics']['min_time']:.2f} seconds")
                print(f"  Max time: {result_data['statistics']['max_time']:.2f} seconds")
                print()

# ----- Application -----

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compare response times between Ollama and ChatGPT')
    parser.add_argument('--prompt', type=str, default="What equipment can help older adults stay safe and avoid falls during winter?.", 
                        help='The prompt to send to both APIs')
    parser.add_argument('--ollama-model', type=str, default="qwen2.5:14b", 
                        help='The Ollama model to use')
    parser.add_argument('--ollama-url', type=str, default="http://localhost:11434",
                        help='The base URL for Ollama API')
    parser.add_argument('--chatgpt-model', type=str, default="gpt-4o-mini", 
                        help='The ChatGPT model to use')
    parser.add_argument('--openai-api-key', type=str, required=True, 
                        help='OpenAI API key')
    parser.add_argument('--runs', type=int, default=5, 
                        help='Number of runs per model')
    parser.add_argument('--output-file', type=str, 
                        help='Output file for JSON results')
    parser.add_argument('--plot-file', type=str, default='response_time_comparison.png',
                        help='Output file for comparison plot')
    
    return parser.parse_args()

def main():
    """Main application entry point"""
    args = parse_arguments()
    
    print("Starting response time evaluation...")
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Number of runs per model: {args.runs}")
    
    # Create services
    ollama_service = OllamaService(model=args.ollama_model, base_url=args.ollama_url)
    chatgpt_service = ChatGPTService(api_key=args.openai_api_key, model=args.chatgpt_model)
    
    # Create output services
    visualizer = TestResultVisualizer()
    repository = TestResultRepository()
    
    # Create and configure test runner
    test_runner = LLMTestRunner(visualizer, repository)
    test_runner.add_service("Ollama", ollama_service)
    test_runner.add_service("ChatGPT", chatgpt_service)
    
    # Run tests
    results = test_runner.run_tests(args.prompt, args.runs)
    
    # Print summary
    test_runner.print_summary(results)
    
    # Create visualization
    try:
        visualizer.plot_comparison(results, args.plot_file)
    except Exception as e:
        print(f"Error creating plot: {e}")
    
    # Save results to file
    repository.save_to_json(results, args.output_file)

if __name__ == "__main__":
    main()