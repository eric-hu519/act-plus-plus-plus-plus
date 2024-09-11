import requests

def test_get_straw():
    response = requests.get("http://192.168.31.109:1116/vision")
    print(response.text)

if __name__ == "__main__":
    test_get_straw()