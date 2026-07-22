import subprocess
import time
import sys
import os
import urllib.request
import json

def test_endpoints():
    print("--- ReviewPulse Backend App Verification ---")
    base_url = "http://127.0.0.1:8500"
    
    # 1. Start uvicorn in a subprocess
    print("Starting FastAPI app on port 8500...")
    env = os.environ.copy()
    # Add project root to python path to avoid ModuleNotFoundError
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8500"],
        env=env
    )
    
    # Wait for uvicorn to boot up by polling the status endpoint
    print("Waiting for server to become ready...")
    server_ready = False
    for i in range(15):
        if proc.poll() is not None:
            break
        try:
            req = urllib.request.Request(f"{base_url}/status")
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200:
                    server_ready = True
                    break
        except Exception:
            pass
        time.sleep(1)
        
    if not server_ready:
        print("[ERROR] uvicorn failed to start or did not become ready in 15 seconds.")
        sys.exit(1)
        
    try:
        # Test 1: Get status
        print("\nTesting /status endpoint...")
        req = urllib.request.Request(f"{base_url}/status")
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            print("[SUCCESS] /status responded with status 200.")
            print("Response:", res_data)
            
        # Test 2: Predict single review
        print("\nTesting /predict endpoint...")
        predict_payload = json.dumps({"text": "The packaging was crushed, but the product inside works perfectly. Mixed feelings."}).encode('utf-8')
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=predict_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            print("[SUCCESS] /predict responded with status 200.")
            print("Response:", res_data)
            assert "sentiment" in res_data
            assert "confidence" in res_data
            
        # Test 3: Get dashboard data
        print("\nTesting /dashboard-data endpoint...")
        req = urllib.request.Request(f"{base_url}/dashboard-data")
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            print("[SUCCESS] /dashboard-data responded with status 200.")
            if res_data.get("status") == "pending_analysis":
                print("Cache is pending analysis. Attempting /analyze...")
                
                # Test 4: Run batch analysis
                print("\nTesting /analyze endpoint...")
                req_analyze = urllib.request.Request(f"{base_url}/analyze", method="POST")
                with urllib.request.urlopen(req_analyze) as response_analyze:
                    res_analyze = json.loads(response_analyze.read().decode())
                    print("[SUCCESS] /analyze completed batch processing.")
                    print("Response:", res_analyze)
                    
                # Re-test /dashboard-data
                print("\nRetesting /dashboard-data endpoint after analysis...")
                with urllib.request.urlopen(req) as response_retest:
                    res_data_retest = json.loads(response_retest.read().decode())
                    print("[SUCCESS] /dashboard-data returns precomputed cached data.")
                    print("Keys in cached data:", list(res_data_retest.keys()))
                    assert "summary" in res_data_retest
                    assert "trends" in res_data_retest
                    assert "keywords" in res_data_retest
            else:
                print("Cache was already populated.")
                print("Keys in cached data:", list(res_data.keys()))
                
        print("\nAll integration checks passed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Verification check failed: {e}")
        # Terminate uvicorn
        proc.terminate()
        proc.wait()
        sys.exit(1)
    finally:
        # 5. Clean up subprocess
        print("\nShutting down uvicorn...")
        proc.terminate()
        proc.wait()
        print("Uvicorn shut down successfully.")

if __name__ == "__main__":
    test_endpoints()
