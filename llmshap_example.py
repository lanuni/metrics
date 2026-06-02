#Remember to install 'pip install llmshap'


from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution
from llmSHAP.llm import OpenAIInterface

data = "In what city is the Eiffel Tower?"
handler = DataHandler(data, permanent_keys={0,3,4})
result = ShapleyAttribution(model=OpenAIInterface(model_name="gpt-4o-mini"),
                            data_handler=handler,
                            prompt_codec=BasicPromptCodec(system="Answer the question briefly."),
                            use_cache=True,
                            num_threads=16,
                            ).attribution()

print("\n\n### OUTPUT ###")
print(result.output)

print("\n\n### ATTRIBUTION ###")
print(result.attribution)

print("\n\n### HEATMAP ###")
print(result.render())
