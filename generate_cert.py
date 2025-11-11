#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成自签名 SSL 证书用于 HTTPS 访问
"""

import os
import sys

def generate_with_openssl():
    """使用 OpenSSL 命令生成证书"""
    import subprocess
    
    print("[证书生成] 使用 OpenSSL 生成证书...")
    
    # 检查 OpenSSL 是否可用
    try:
        subprocess.run(["openssl", "version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[错误] 未找到 OpenSSL，请先安装 OpenSSL")
        print("Windows: https://slproweb.com/products/Win32OpenSSL.html")
        print("或使用: pip install pyopenssl 然后运行此脚本")
        return False
    
    # 生成证书
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-nodes", "-out", "cert.pem", "-keyout", "key.pem",
        "-days", "365",
        "-subj", "/C=CN/ST=State/L=City/O=AIGlasses/CN=localhost"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("[成功] 证书已生成:")
        print("  - cert.pem (证书文件)")
        print("  - key.pem (私钥文件)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[错误] 生成证书失败: {e}")
        return False

def generate_with_pyopenssl():
    """使用 PyOpenSSL 库生成证书"""
    try:
        from OpenSSL import crypto
    except ImportError:
        print("[错误] 未安装 pyOpenSSL")
        print("请运行: pip install pyopenssl")
        return False
    
    print("[证书生成] 使用 PyOpenSSL 生成证书...")
    
    # 创建密钥对
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 4096)
    
    # 创建自签名证书
    cert = crypto.X509()
    cert.get_subject().C = "CN"
    cert.get_subject().ST = "State"
    cert.get_subject().L = "City"
    cert.get_subject().O = "AIGlasses"
    cert.get_subject().OU = "Development"
    cert.get_subject().CN = "localhost"
    
    # 添加 SAN (Subject Alternative Name) 支持多个域名
    cert.add_extensions([
        crypto.X509Extension(
            b"subjectAltName",
            False,
            b"DNS:localhost,DNS:*.local,IP:127.0.0.1,IP:0.0.0.0"
        ),
    ])
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # 1年有效期
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')
    
    # 保存证书和密钥
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    print("[成功] 证书已生成:")
    print("  - cert.pem (证书文件)")
    print("  - key.pem (私钥文件)")
    return True

def main():
    print("=" * 60)
    print("AI盲人眼镜系统 - SSL 证书生成工具")
    print("=" * 60)
    print()
    
    # 检查是否已存在证书
    if os.path.exists("cert.pem") and os.path.exists("key.pem"):
        print("[警告] 证书文件已存在")
        response = input("是否覆盖现有证书? (y/N): ").strip().lower()
        if response != 'y':
            print("[取消] 保留现有证书")
            return
        print()
    
    # 尝试生成证书
    success = False
    
    # 方法1: 尝试使用 PyOpenSSL
    try:
        success = generate_with_pyopenssl()
    except Exception as e:
        print(f"[PyOpenSSL 失败] {e}")
        print()
    
    # 方法2: 如果 PyOpenSSL 失败，尝试使用 OpenSSL 命令
    if not success:
        success = generate_with_openssl()
    
    if success:
        print()
        print("=" * 60)
        print("[完成] 证书生成成功！")
        print()
        print("下一步:")
        print("1. 重启服务器: python app_main.py")
        print("2. 使用 HTTPS 访问: https://[服务器IP]:8081/mobile")
        print("3. 首次访问时接受证书警告")
        print()
        print("注意:")
        print("- 自签名证书会显示\"不安全\"警告，这是正常的")
        print("- 点击\"高级\" → \"继续访问\"即可")
        print("- iOS 用户可能需要在设置中信任证书")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("[失败] 无法生成证书")
        print()
        print("替代方案:")
        print("1. 使用 ngrok: ngrok http 8081")
        print("2. 手动安装 OpenSSL 后重试")
        print("3. 使用 pip install pyopenssl 后重试")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()