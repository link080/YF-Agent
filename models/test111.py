from inference import predict_intent

tests = ['上海美居酒店今天，大床', '上海美居', '好的', '确认', '就这样吧', '能便宜点吗', '有停车场吗']
for t in tests:
    r = predict_intent(t)
    intent = r['intent']
    conf = r['confidence']
    print(f"{t} -> {intent} ({conf:.2%})")
