AttributeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 88, in exec_func_with_error_handling
    result = func()
             ^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 579, in code_to_exec
    exec(code, module.__dict__)
File "/mount/src/poc/LLM/rag_md.py", line 200, in <module>
    main()
File "/mount/src/poc/LLM/rag_md.py", line 62, in main
    text_chunks = get_text_chunks(files_text)  # 텍스트 청크로 분할
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/poc/LLM/rag_md.py", line 158, in get_text_chunks
    header_chunks = header_splitter.split_documents(texts)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
