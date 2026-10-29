/Users/andy_mac/Documents/CodeSpace/BackEnd/andyPython/src/qianWenPro.py
这个文件是可以正常运行的，
1-请再写一个 流式输出的版本，qianWenProStream.py
2-请再写一个 流式输出的版本，qianWenProStream1.py，我可以直接运行 和他多轮对话，
2-请再写一个 流式输出的版本，qianWenProStream2.py，我可以使用 stringboot 项目进行 fastapi 调用。


有一个问题需要你决定： qwen3.7-plus 默认开着思考模式。实测开着时第一个字要 8 秒左右，token 消耗约是关掉的 14 倍（486 对 35）；关掉后约 1.4 秒出字。原脚本也是开着的，为了不改原来的效果，我保留了思考模式，并把思考过程也流式输出（命令行里是灰色），这样不会干等。每个文件顶部有 ENABLE_THINKING 开关，想要更快、更省钱就改成 False。

Spring Boot 接口约定：
- 请求： {"messages":[{"role":"user","content":"..."}]}。多轮对话由 Java 端把历史一起传过来，Python 服务不保存会话，可以直接部署多个实例。
- 返回： SSE 流，每条 data 是一行 JSON：
  - reasoning：思考过程，可以展示也可以忽略
  - delta：回答的增量文字
  - done：结束，带 token 用量
  - error：调用出错
- 状态码： 参数不合法返回 422；单实例同时超过 100 个流返回 429。
- 断开处理： Java 端中途断开，服务会关闭到千问的连接，不会泄漏，已实测。

webClient.post().uri("/api/v1/chat/stream")
    .contentType(MediaType.APPLICATION_JSON).accept(MediaType.TEXT_EVENT_STREAM)
    .header("X-Request-ID", requestId).bodyValue(chatStreamRequest)
    .retrieve().bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<String>>() {})
    .map(ServerSentEvent::data); // 按 JSON 里的 type 分别处理

测试情况：
- 写了 tests/test_qianWenProStream.py，11 个用例全部通过，覆盖正常流式、出错、参数校验、429、历史截断等，不调用真实接口。
- 用真实接口验证过：多轮对话记得名字，clear 后不记得；SSE 多轮正常；断开连接后资源正常释放。

其他：
- 在 pyproject.toml 里加了 fastapi 和 uvicorn 依赖。
- 原来的 qianWenPro.py 我没改，它会把 API key 打印到控制台；新写的三个文件都不打印。
- 