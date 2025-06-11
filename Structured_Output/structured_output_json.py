from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')


# schema
json_schema = {
    "title" : "Review",
    "type" : "object",
    "properties": {
        "key_themes" : {
            "type" : "array",
            "items" : {
                "type" : "string"
            },
            "description" : "write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type" : "string",
            "description": "A brief summary of the review"
        },
        "sentiment":{
            "type": "string",
            "enum": ["pos","neg"],
            "description": "Return sentiment of the review either negative, positive or neutral"
        },
        "pros":{
            "type" : ["array","null"],
            "items": {
                "type":"string"
            },
            "description" : "write down the pros inside a list"
        },
        "cons":{
            "type" : ["array","null"],
            "items": {
                "type":"string"
            },
            "description" : "write down the cons inside a list"
        },
        "name": {
        "type": ["string","null"],
        "description": "write the name of the reviewer"
        }
    },
    "required" : ["key_themes","summary","sentiment"]
}
    

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""The iPhone 15 is a well-rounded upgrade that brings meaningful changes like the Dynamic Island, a new 48MP main camera, and the long-awaited USB-C charging port. Powered by the A16 Bionic chip, it delivers smooth performance and improved power efficiency. The camera captures sharper, more detailed photos, and the refreshed design with slightly curved edges offers a more comfortable grip. However, the lack of a 120Hz ProMotion display and no dedicated telephoto lens might disappoint some users, especially considering the price point. For those upgrading from an iPhone 12 or earlier, the iPhone 15 feels like a significant step up. But if you’re coming from an iPhone 13 or 14, the improvements may feel incremental. Overall, it’s a polished, future-ready device that balances performance, design, and features well. Review by mkbhd""")

print(result)