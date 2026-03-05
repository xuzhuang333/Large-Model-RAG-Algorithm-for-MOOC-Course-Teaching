import zipfile
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger("XMind_Debugger")


def debug_xmind_parsing(xmind_path):
    logger.info(f"Inspecting XMind file: {xmind_path}")

    if not os.path.exists(xmind_path):
        logger.error("File does not exist. Please check the path.")
        return

    try:
        with zipfile.ZipFile(xmind_path, 'r') as xmind_zip:
            file_list = xmind_zip.namelist()
            logger.info(f"Files inside the XMind archive: {file_list}")

            if 'content.json' in file_list:
                logger.info("Found content.json. Attempting to parse...")
                with xmind_zip.open('content.json') as f:
                    content = json.loads(f.read().decode('utf-8'))

                    # Usually the root array has one dictionary
                    if isinstance(content, list) and len(content) > 0:
                        root_topic = content[0].get('rootTopic', {})
                        title = root_topic.get('title', 'NO_TITLE')
                        children = root_topic.get('children', {})

                        print(f"\n--- Parsing Success ---")
                        print(f"Root Title: {title}")
                        print(f"Has Children: {'attached' in children}")
                        print(
                            f"Raw Root Node Data: {json.dumps(root_topic, ensure_ascii=False, indent=2)[:500]}...")  # Print first 500 chars
                    else:
                        logger.warning("content.json structure is not as expected.")

            elif 'content.xml' in file_list:
                logger.error("This is an old XMind 8 format (content.xml). Our script only handles content.json!")
            else:
                logger.error("Neither content.json nor content.xml found in the archive.")

    except zipfile.BadZipFile:
        logger.error("This file is not a valid ZIP/XMind archive. It might be corrupted.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # 替换为你 E 盘中 4.4.1 XMind 文件的真实物理路径
    test_xmind_path = r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学\【第4周】程序的控制结构\4.1 程序的分支结构\4.1.1 单元开篇\4.1.1 单元开篇.xmind"

    debug_xmind_parsing(test_xmind_path)