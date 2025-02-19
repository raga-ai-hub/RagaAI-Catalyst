# Prompt Management

The Prompt Management feature in RagaAI Catalyst allows you to efficiently manage, retrieve, and use prompts in your projects. 

## Table of Contents
1. [Library Detail](#library-detail)
2. [Error Handling](#error-handling)
3. [FAQs](#faqs)

## Library Detail

### 1. Initialize RagaAICatalyst and PromptManager

First, set up your RagaAICatalyst instance and create a PromptManager for your project:

```python
from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.prompt_manager import PromptManager

catalyst = RagaAICatalyst(
access_key="your_access_key",
secret_key="your_secret_key",
base_url="https://your-api-base-url.com/api"
)
```

Create a PromptManager for your project:

```python
project_name = "your-project-name"
prompt_manager = PromptManager(project_name)
```

### 2. List Available Prompts

```python
prompts = prompt_manager.list_prompts()
print("Available prompts:", prompts)
```

### 3. List Prompt Versions

```python
prompt_name = "your_prompt_name"
versions = prompt_manager.list_prompt_versions(prompt_name)
```

### 4. Get a Prompt Object

Retrieve a prompt object by name:

```python
prompt_name = "your_prompt_name"
prompt = prompt_manager.get_prompt(prompt_name)

```

Retrieve a specific prompt object by name and version:

```python
prompt_name = "your_prompt_name"
version = "your_version"
prompt = prompt_manager.get_prompt(prompt_name, version)
```

### 5. Get Prompt Variables

```python
prompt_variables = prompt.get_variables()
print("prompt_variables: ",prompt_variables)
```

### 6. Get Prompt Versions

```python
prompt_versions = prompt.list_prompt_versions()
print("prompt_versions: ",prompt_versions)
```

### 7. Compile Prompt

Once you have a prompt, you can compile it with variables:

```python
compiled_prompt = prompt.compile(query="What's the weather?", context="sunny", llm_response="It's sunny today")
print("Compiled prompt:", compiled_prompt)
```

### 8. Get Parameters

```python
parameters = prompt.get_parameters()
print("parameters: ",parameters)
```



## Error Handling

### 1. Project Not Found

If the project you are trying to access does not exist, the `PromptManager` will raise a `ValueError`:

```python
prompt_manager = PromptManager("non_existent_project")

# Error: Project not found. Please enter a valid project name
```

### 2. Prompt Not Found

If the prompt you are trying to access does not exist, the `get_prompt` method will raise a `ValueError`:

```python
prompt = prompt_manager.get_prompt("non_existent_prompt")

# Error: Prompt not found. Please enter a valid Prompt name
```

### 3. Prompt Version Not Found

If the prompt version you are trying to access does not exist, the `get_prompt` method will raise a `ValueError`:

```python
prompt = prompt_manager.get_prompt("your_prompt_name", "non_existent_version")

# Error: Version not found. Please enter a valid version name
```

### 4. Missing Variables in Compile

If the variables you are trying to compile the prompt with are not found, the `compile` method will raise a `ValueError`:

```python
prompt = prompt_manager.get_prompt("your_prompt_name", "your_version")
prompt.get_variables()
compiled_prompt = prompt.compile(query="What's the weather?")

# Error: Missing variable(s): context, llm_response
```

### 5. Extra Variables in Compile

If the variables you are trying to compile the prompt with are not found, the `compile` method will raise a `ValueError`:

```python
prompt = prompt_manager.get_prompt("your_prompt_name", "your_version")
compiled_prompt = prompt.compile(query="What's the weather?", context="sunny", llm_response="It's sunny today", expected_response="The weather is sunny")

# Error: Extra variable(s) provided: expected_response
```

### 6. Types of variable not str

If the variables you are trying to compile the prompt with are not 'str', the `compile` method will raise a `ValueError`:

```python
prompt = prompt_manager.get_prompt("your_prompt_name", "your_version")
compiled_prompt = prompt.compile(query=True, context="sunny", llm_response="It's sunny today")

# Error: Value for variable 'query' must be a string, not bool
```


## FAQs

### 1. How do I get the list of prompts in a project?

You can get the list of prompts in a project by using the `list_prompts()` method in the `PromptManager`. This method allows you to retrieve the list of prompts in a project.

### 2. How do I get the versions of a prompt?

You can get the versions of a prompt by using the `list_prompt_versions(prompt_name)` method in the `PromptManager`. This method allows you to retrieve the versions of a prompt.

### 3. How do I get the default version of a prompt?

You can get the default version of a prompt by using the `get_prompt(prompt_name)` method in the `PromptManager`. This method allows you to retrieve the default version of a prompt. Then you can use `compile` method to get the prompt with default variables.

### 4. How do I get the specific versions of a prompt?

You can get the versions of a prompt by using the `get_prompt(prompt_name, version)` method in the `PromptManager`. This method allows you to retrieve the versions of a prompt. Then you can use `compile` method to get the prompt with default variables.


### 5. How do I get the variables of a prompt?

You can get the variables of a prompt by using the `get_variables()` method. This method allows you to retrieve the variables of a prompt.

### 6. How do I get my parameters?

You can get the parameters of a prompt by using the `get_parameters()` method. This method allows you to retrieve the parameters of a prompt.
