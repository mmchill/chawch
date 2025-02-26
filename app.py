from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
import os
import argparse
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI

@tool
def my_custom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Launch agent with Gradio UI")
    parser.add_argument("--hf_token", type=str, help="Hugging Face API token")
    parser.add_argument("--share", action="store_true", help="Create a shareable link", default=False)
    parser.add_argument("--model_id", help="The agent model id (HF id)", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("Warning: No Hugging Face token provided. Some features may not work.")
        print("Set with --hf_token or HF_TOKEN environment variable.")
    else:
        os.environ["HF_TOKEN"] = hf_token
        print("Hugging Face token set successfully.")

    final_answer = FinalAnswerTool()
    
    # If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
    # model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 
    
    model = HfApiModel(
        max_tokens=2096,
        temperature=0.5,
        model_id=args.model_id,
        custom_role_conversions=None,
        token=hf_token
    )
    
    # Load image generation tool (with token)
    try:
        image_generation_tool = load_tool("agents-course/text-to-image", 
                                          trust_remote_code=True,
                                          token=hf_token)
        tools = [final_answer, image_generation_tool, 
                 get_current_time_in_timezone, my_custom_tool]
        print("Image generation tool loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load image generation tool: {e}")
        tools = [final_answer, get_current_time_in_timezone, my_custom_tool]
    
    # Load prompt templates
    with open("prompts.yaml", 'r') as stream:
        prompt_templates = yaml.safe_load(stream)
    
    # Initialize agent
    agent = CodeAgent(
        model=model,
        tools=tools,  # Add your tools here (don't remove final answer)
        max_steps=6,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=prompt_templates
    )
    
    # Launch Gradio UI
    GradioUI(agent).launch(share = args.share)

if __name__ == "__main__":
    main()