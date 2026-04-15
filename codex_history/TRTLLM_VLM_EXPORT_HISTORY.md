# TRTLLM VLM Export History

## 2026-04-14

### 시도해본것
- `venvs/trtllm`에서 `TensorRT-LLM` import와 `trtllm-build` 실행 가능 여부를 확인했다.
- `TensorRT-LLM 1.2.0` 패키지 내부에서 `Qwen3VLForConditionalGeneration` 지원 경로를 확인했다.
- `nvidia/Alpamayo-1.5-10B`와 `nvidia/Cosmos-Reason2-8B`의 로컬 Hugging Face 캐시를 비교해 체크포인트 구조를 조사했다.
- Alpamayo 체크포인트에서 `vlm.*`, `expert.*`, `action_in_proj.*`, `action_out_proj.*` 가중치가 분리되어 있는지 확인했다.
- Alpamayo의 `vlm.*` 가중치를 Qwen3-VL HF 포맷으로 재포장하는 스크립트 초안을 `tools/export_alpamayo_vlm_to_qwen3vl.py`로 추가했다.
- 변환 스크립트를 dry-run으로 실행해 shard 분포와 vocab 확장 계획을 검증했다.
- 변환 스크립트를 실제 실행해 `/home/jys/workspace/alpamayo_qwen3vl_hf`에 Qwen3-VL 파생 체크포인트를 생성했다.
- 생성된 출력 디렉터리에 대해 tokenizer id, config, safetensors index, TRT-LLM model class 인식 여부를 검증했다.
- `TensorRT-LLM llmapi`와 `_tensorrt_engine.LLM` 경로를 읽어, 로컬 HF 체크포인트에서 바로 엔진을 빌드하는 진입점을 확인했다.
- repacked VLM 체크포인트를 실제 엔진 디렉터리로 저장하기 위한 빌드 스크립트 `tools/build_trtllm_vlm_engine.py`를 추가했다.
- `tools/build_trtllm_vlm_engine.py`로 `/home/jys/workspace/alpamayo_qwen3vl_hf -> /home/jys/workspace/alpamayo_qwen3vl_trtllm_engine` 실제 엔진 빌드를 시도했다.
- `Qwen3VLForConditionalGeneration`를 full VLM으로 태우는 경로가 실제로 가능한지 확인하기 위해 `tensorrt_llm.models.MODEL_MAP`, `qwen/config.py`, `qwen/model.py`, `qwen/convert.py`를 다시 추적했다.
- `Qwen3-VL`의 language model만 떼어 `Qwen3ForCausalLM` HF 체크포인트로 재포장하는 스크립트 `tools/export_qwen3vl_text_decoder.py`를 추가했다.
- text-only 파생 체크포인트를 `/home/jys/workspace/alpamayo_qwen3_text_hf`로 생성했다.
- `_torch/auto_deploy`, `_torch/pyexecutor`, `_torch/auto_deploy/transform/*`를 조사해 new API 쪽에 직접 TensorRT `.plan` 엔진을 저장하는 진입점이 있는지 확인했다.
- `tools/build_trtllm_vlm_engine.py`에 local architecture guard를 넣어, 현재 설치본에서 full `Qwen3VLForConditionalGeneration`를 잘못된 TensorRT backend API로 바로 태우지 않도록 수정했다.
- `tools/build_qwen3_text_decoder_engine.py`를 추가해 `Qwen3-VL -> Qwen3 text decoder 추출 -> TensorRT backend build` 순서를 하나의 올바른 진입점으로 묶었다.

### 실패한점
- 샌드박스 안에서는 `mpi4py`가 OpenMPI 소켓 초기화 제한 때문에 실패해서 `tensorrt_llm` import 검증이 바로 되지 않았다.
- 기본 환경 변수만으로는 `libcublasLt.so.13` 등 CUDA 13/TensorRT 라이브러리를 찾지 못해 `tensorrt_llm` import가 실패했다.
- `nvidia/Alpamayo-1.5-10B` 전체 체크포인트는 `architectures=["Alpamayo1_5"]`라서 TRT-LLM auto model 경로에 그대로 넣을 수 없다.
- Alpamayo 스냅샷에는 tokenizer 파일이 없어서 base VLM tokenizer를 재구성해서 써야 한다.
- 처음 생성된 `model.safetensors.index.json`의 `metadata.total_parameters`는 weight key 개수로 기록되어 있어 실제 파라미터 수와 달랐다.
- `trtllm-build` 구형 CLI는 TRT-LLM checkpoint 포맷 기준이라, 이번처럼 로컬 HF repacked checkpoint를 직접 넣는 주 경로로 쓰기에는 맞지 않았다.
- 실제 TensorRT backend 엔진 빌드 시 `TrtLlmArgs`가 HF `Qwen3VLForConditionalGeneration`을 지원하지 않는다고 판단해 생성자 단계에서 중단됐다.
- 즉 설치된 TRT-LLM 안에 `_torch` 기준 `Qwen3VLModel`은 있지만, `tensorrt_llm.models.automodel` 기반 TensorRT backend 지원 목록에는 아직 연결되지 않았다.
- 실행 시점에 `SM 12.x requires CUDA >= 12.9` 경고도 계속 발생해, 이후 support gate를 우회하더라도 CUDA 조합 검증이 추가로 필요하다.
- `Qwen3VLForConditionalGeneration`를 단순히 `MODEL_MAP`에 추가하는 것만으로는 충분하지 않다. 현재 TensorRT backend의 Qwen 구현은 old `QWenForCausalLM`이며, 이 클래스는 full vision tower를 포함하지 않는다.
- old backend의 `QWenConfig.from_hugging_face()`도 top-level `model_type=qwen3_vl`을 직접 처리하지 못한다. 즉 genuine full `Qwen3-VL`은 단순 alias/문자열 패치로 해결되지 않는다.
- new `_torch` 계층은 존재하지만, 설치된 패키지 안에서 확인된 경로는 `torch.export`, `torch.compile`, `pyexecutor`, `auto_deploy` 중심이었다. 즉 이 계층에서 바로 TensorRT `.plan` 엔진 파일을 저장하는 공개 진입점은 확인되지 않았다.
- API를 잘못 선택하면 full VLM과 text-only decoder가 같은 빌더로 섞이기 쉬웠다. 기존 스크립트는 이 경계를 코드로 강제하지 않아 사용자가 잘못된 경로를 다시 밟을 수 있었다.

### 성공한점
- `TensorRT-LLM` 안에 `_torch` 경로 기준 `Qwen3VLForConditionalGeneration` 지원이 실제로 존재함을 확인했다.
- 올바른 `LD_LIBRARY_PATH`를 주면 `tensorrt_llm` import와 `trtllm-build --help`가 정상 동작함을 확인했다.
- Alpamayo 체크포인트의 VLM 가중치가 `vlm.model.visual.*`, `vlm.model.language_model.*`, `vlm.lm_head.weight` 구조로 저장되어 있어 Qwen3-VL 파생 체크포인트로 재포장 가능하다는 점을 확인했다.
- base VLM `nvidia/Cosmos-Reason2-8B`가 `Qwen3VLForConditionalGeneration` 아키텍처이며, Alpamayo가 이 모델의 vocab만 확장한 구조라는 점을 확인했다.
- `tools/export_alpamayo_vlm_to_qwen3vl.py`로 실제 VLM 파생 체크포인트를 4개 safetensors shard와 HF tokenizer/processor 파일로 생성했다.
- 생성 결과의 tokenizer 크기 `155697`, `traj_token_start_idx=151669`, trajectory special token id가 Alpamayo config와 일치함을 확인했다.
- 생성 결과의 HF config가 `Qwen3VLConfig`로, TRT-LLM `_torch` 경로에서는 `Qwen3VLModel`로 인식됨을 확인했다.
- 출력 체크포인트의 safetensors index에서 주요 키(`model.visual.patch_embed.proj.weight`, `model.language_model.embed_tokens.weight`, `lm_head.weight`) 매핑이 정상임을 확인했다.
- `TensorRT` backend의 `LLM` 클래스가 내부적으로 HF checkpoint를 로드하고 엔진을 build/save 하는 경로를 사용한다는 점을 확인했다.
- `repacked HF dir -> TensorRT-LLM engine dir`를 직접 태울 수 있는 재현 가능한 스크립트 엔트리포인트를 준비했다.
- 실제 빌드 시도를 통해 실패 지점을 “가중치/토크나이저 문제가 아니라 TensorRT backend의 모델 지원 게이트”로 좁혔다.
- 빌드 명령, 환경 변수, 출력 경로, 실패 로그가 재현 가능한 형태로 정리됐다.
- old TensorRT backend의 Qwen 경로가 실제로는 `qwen2_vl`까지의 multimodal 보정만 일부 가지고 있고, `qwen3_vl` full model class는 포함하지 않는다는 점을 코드 수준에서 확인했다.
- `Qwen3-VL` 안의 `language_model` 가중치를 `Qwen3ForCausalLM` 형식으로 재포장하는 우회 경로를 마련했다.
- text-only 파생 체크포인트 `/home/jys/workspace/alpamayo_qwen3_text_hf`에는 `Qwen3ForCausalLM` config와 renamed text weights(`model.*`, `lm_head.weight`)가 정상적으로 생성됐다.
- new `_torch` 경로의 성격을 분리했다. 이쪽은 “새 모델 구현/graph export/runtime 최적화” 계층이지, 현재 설치본 기준 “TensorRT 엔진 저장 API”로 바로 이어지는 계층은 아니었다.
- 현재 설치본에서 실제로 맞는 API 경계를 코드에 반영했다. full `Qwen3-VL`은 guard로 차단하고, `Qwen3` text decoder는 전용 래퍼 스크립트로 올바른 변환+build 순서를 타도록 정리했다.

### 보완하면 좋을만한점
- 변환 스크립트 실행 후 생성된 tokenizer/processor 파일과 Alpamayo inference 경로의 실제 token id가 1:1로 맞는지 추가 검증이 필요하다.
- 재포장된 Qwen3-VL 체크포인트를 TRT-LLM `_torch` loader로 실제 로드하는 검증 루틴을 붙이면 실패 지점을 더 빨리 찾을 수 있다.
- 추후에는 `expert`와 `action_*` 쪽도 별도 TRT-LLM/엔진 경로로 분리하는 스크립트를 추가하는 편이 좋다.
- 변환 스크립트의 진행 출력은 shard 단위까지만 있어서, 장시간 실행 시 tensor 수나 현재 shard 상태를 더 자세히 출력하면 추적이 쉬워진다.
- 실제 `trtllm-build` 호출 전, repacked checkpoint를 `ModelConfig.from_pretrained()`와 weight load 경로까지 태우는 검증을 추가하면 안전하다.
- VLM engine build 스크립트에는 아직 모델별 입력 프로파일 자동 조정 로직이 없어서, 실제 워크로드에 맞춘 `max_input_len`, `max_seq_len`, `max_num_tokens` 튜닝이 추가되면 좋다.
- 현재 설치본에서는 `_torch` 지원과 TensorRT backend 지원 목록이 분리되어 있으므로, TRT-LLM 버전 업 또는 로컬 패치로 `Qwen3VLForConditionalGeneration`이 TensorRT backend 경로에 연결되는지 먼저 확인할 필요가 있다.
- support gate가 풀린 뒤에는 같은 스크립트로 바로 재시도할 수 있으니, 다음 단계는 빌드 로직 추가보다 설치본/소스 패치 검증이 우선이다.
- full `Qwen3-VL`을 정말 TensorRT backend 엔진 하나로 가져가려면, `_torch/models/modeling_qwen3vl.py` 계열이 old TensorRT backend build 경로와 연결되는 별도 업스트림 지원이 필요하다.
- 현 설치본에서는 실용적인 경로가 `vision`과 `text decoder`를 분리하는 방식이므로, 이후 작업도 그 분해를 기준으로 진행하는 편이 리스크가 낮다.
- 만약 upstream에 `_torch Qwen3VL -> TensorRT engine` 브리지가 추가된 새 버전이 있다면, 그 버전으로 올리는 것이 소스 패치보다 빠를 수 있다.
- 이후 upstream 지원이 추가되면 full VLM build guard는 해제하거나, 별도 `full_vlm` builder를 추가하면 된다.

### 다음스텝
- `Qwen3VLForConditionalGeneration` full model을 one-shot으로 빌드하는 new public API는 현재 설치본에서 확인되지 않았으므로, 이후 VLM 쪽은 `vision encoder` 엔진과 `Qwen3` text decoder 엔진으로 분리해 진행한다.
- `tools/build_qwen3_text_decoder_engine.py`를 기준 진입점으로 삼아 text decoder 엔진 빌드가 실제 완료되는지 다시 확인한다.
- vision 쪽은 TRT-LLM old backend가 아니라 별도 multimodal/vision export 경로를 찾아 붙인다.
- backbone 쪽이 정리되면 `expert`, `action_in_proj`, `action_out_proj`를 별도 엔진화 대상으로 분리한다.

## 2026-04-14 (LLaVA Triton Guide 확인)

### 시도해본것
- `/home/jys/Downloads/llava_trtllm_guide.md`를 읽어 Triton과 TensorRT-LLM 멀티모달 배포 절차를 확인했다.
- 가이드가 요구하는 핵심 단계가 `언어 모델용 TRT-LLM engine`과 `visual encoder용 TensorRT engine`의 분리 빌드인지 확인했다.
- 로컬 설치본 `venvs/trtllm` 안에서 `multimodal_builder.py`, `multimodal_model_runner.py`를 조사해 현재 패키지에 남아 있는 멀티모달 비전 엔진 빌드 경로를 확인했다.
- `multimodal_builder.py`의 지원 모델 목록과 `qwen2_vl` 전용 visual engine builder 존재 여부를 확인했다.

### 실패한점
- LLaVA 가이드는 `v0.9.0` 계열 튜토리얼이라 현재 로컬 `TensorRT-LLM 1.2.0`과 API가 완전히 일치하지 않는다.
- 로컬 설치본에는 LLaVA/Qwen2-VL용 visual builder는 있지만, 같은 계열의 `qwen3_vl` visual builder는 확인되지 않았다.
- 따라서 문서의 빌드 명령을 `Qwen3-VL`에 그대로 옮겨 적용할 수는 없다.

### 성공한점
- 문서상으로도 멀티모달 모델은 `visual_encoder.engine`과 `TRT-LLM language engine`으로 나눠 배포하는 방식이 공식 튜토리얼에 포함되어 있음을 확인했다.
- 로컬 설치본에도 `tensorrt_llm.tools.multimodal_builder`가 존재하고, `qwen2_vl`용 visual engine build 함수 `build_qwen2_vl_engine(args)`가 포함되어 있음을 확인했다.
- 로컬 런타임 `multimodal_model_runner.py`에도 `qwen2_vl` 멀티모달 실행 경로가 남아 있어, 설치본이 “비전 엔진 + 언어 엔진 분리” 구조를 실제로 전제로 하고 있음을 재확인했다.
- 이로써 Alpamayo/Qwen3-VL에 대해서도 full one-shot engine이 막히면 `vision encoder 별도 엔진 + text decoder TRT-LLM 엔진`으로 가는 전략이 문서/로컬 코드 모두와 정합적이라는 근거를 확보했다.

### 보완하면 좋을만한점
- `qwen2_vl` visual engine builder가 어떤 HF class와 입력 시그니처를 사용하는지 더 자세히 뜯어보면 `qwen3_vl`용 포팅 범위를 구체화할 수 있다.
- Triton이 필요한지, 아니면 우선 로컬 inference runtime에서 `vision encoder engine + text decoder engine` 조합만 검증할지 의사결정을 먼저 하면 구현 범위를 줄일 수 있다.
- `Qwen3VL`의 `_torch` vision 구현과 `multimodal_builder.py`의 `qwen2_vl` ONNX/TRT 빌드 패턴을 비교해서, 로컬 커스텀 visual builder 초안을 만드는 것이 다음 생산적인 단계다.

### 다음스텝
- `multimodal_builder.py`의 `build_qwen2_vl_engine(args)`를 기준으로 `Qwen3-VL` visual encoder를 별도 엔진으로 내릴 수 있는지 구조를 분석한다.
- `Qwen3-VL` text decoder는 기존 `tools/build_qwen3_text_decoder_engine.py` 경로로 계속 밀고, vision 쪽은 별도 builder 설계로 분기한다.
- 이후 필요하면 Triton은 마지막 서빙 계층으로 붙이고, 그 전에는 로컬에서 `vision engine + text decoder engine` 조합을 먼저 검증한다.

## 2026-04-14 (PyTorch backend 스모크 테스트)

### 시도해본것
- `TensorRT-LLM`의 `serve.py`, `llm.py`, `llm_args.py`, `inputs/data.py`, `inputs/multimodal.py`, `modeling_qwen3vl.py`를 읽어 `PyTorch backend` 실제 진입점이 `tensorrt_llm.LLM(..., backend="pytorch")`임을 확인했다.
- `Qwen3VLInputProcessorBase`가 `prompt`와 `multi_modal_data={"image": [PIL.Image]}` 형태의 입력을 받는다는 점을 확인했다.
- `tools/run_trtllm_pytorch_qwen3vl_smoke.py`를 추가해 repacked `Qwen3-VL` 체크포인트를 `TensorRT-LLM PyTorch backend`에서 직접 초기화하고 생성까지 시도하는 스모크 테스트 경로를 만들었다.
- 스모크 스크립트에 `disable_overlap_scheduler` 옵션을 추가해 scheduler 영향도 비교할 수 있게 했다.
- `/home/jys/workspace/alpamayo_qwen3vl_hf`와 샘플 이미지 `grace_hopper.jpg`를 사용해 실제 GPU 초기화와 멀티모달 생성 테스트를 수행했다.
- 비교군으로 텍스트 전용 생성 테스트도 수행했다.
- `nvidia-smi`로 GPU 메모리 사용 현황을 확인했다.

### 실패한점
- 멀티모달 생성 테스트는 모델 초기화와 KV cache 할당까지는 통과했지만, 첫 생성 응답이 오래 지연되며 종료까지 확인하지 못했다.
- 텍스트 전용 테스트는 `Executor creation failed due to insufficient GPU memory`로 실패했다.
- 실패 시점의 GPU는 총 31.33 GiB 중 약 23.4 GiB가 이미 점유 중이었고, 특히 `venvs/trtllm/bin/python` 프로세스 하나가 약 20.8 GiB를 잡고 있었다.
- 따라서 이번 실패는 “Qwen3-VL PyTorch backend 미지원”보다 “남아 있는 worker/실행 프로세스로 인한 VRAM 부족” 영향이 더 컸다.

### 성공한점
- `TensorRT-LLM` 로컬 설치본에서 `Qwen3-VL`을 `PyTorch backend`로 태우는 최소 실행 스크립트를 만들었다.
- 실제 실행 로그에서 `Using LLM with PyTorch backend`가 출력되었고, repacked `Qwen3-VL` safetensors shard 4개와 동시 weight load가 정상적으로 진행되었다.
- 멀티모달 테스트에서 `Model init total -- 24.18s`, `Model init total -- 11.71s` 로그까지 확인했다. 즉 `Qwen3-VL` 모델 로드와 executor 초기화 상당 부분은 실제로 진행된다.
- 멀티모달 테스트에서 KV cache block 할당과 attention workspace 경고까지 확인되어, 최소한 “지원 목록 진입 자체가 안 된다” 수준의 문제는 아님을 확인했다.
- 현재 blocker가 API 미지원보다 메모리/런타임 상태 쪽이라는 근거를 확보했다.

### 보완하면 좋을만한점
- 이전 테스트에서 남은 TRT-LLM worker 프로세스를 정리한 뒤 같은 스모크 테스트를 다시 실행해, 멀티모달 생성이 실제 완료되는지 확인해야 한다.
- `max_num_tokens`, `max_seq_len`, `cuda_graph_config`, `free_gpu_memory_fraction` 등을 더 보수적으로 줄인 저메모리 preset을 스크립트에 추가하면 재현성이 좋아진다.
- 필요하면 `MultimodalEncoder` 전용 스모크 테스트도 추가해, full generation과 vision-only 경로를 분리 검증하는 것이 좋다.
- 장기적으로는 Alpamayo 통합 전에 `self.vlm`만 `TensorRT-LLM PyTorch backend`로 교체하는 어댑터를 별도로 만드는 편이 안전하다.

### 다음스텝
- 먼저 GPU에 남아 있는 TRT-LLM Python 프로세스를 정리해 VRAM을 회수한 뒤, 같은 스모크 테스트를 다시 실행한다.
- 재실행 시 `max_num_tokens`, `max_seq_len`을 더 줄여 저메모리 조건에서 초기화/생성 완료 여부를 확인한다.
- 멀티모달 생성이 완료되면 그다음 단계로 Alpamayo `self.vlm`을 `TensorRT-LLM PyTorch backend` 래퍼로 교체하는 방향을 설계한다.

## 2026-04-15 (Codex Harness 반영)

### 시도해본것
- 기존 작업 히스토리 markdown 파일들을 `/home/jys/alpamayo1.5/codex_history/`로 이동했다.
- Codex 하네스 설계 메모를 `/home/jys/alpamayo1.5/codex_md/CODEX_HARNESS_ENGINEERING.md`로 정리했다.
- 하네스 메모를 자동 적용 규칙으로 압축해 repo root `/home/jys/alpamayo1.5/AGENTS.md`를 새로 작성했다.

### 실패한점
- 없음.

### 성공한점
- Codex가 자동 수집할 수 있는 root `AGENTS.md`를 추가했다.
- `AGENTS.md`에 읽기 순서, 히스토리 보관 위치, 검색/편집 원칙, 금지 디렉터리, TensorRT-LLM 작업 원칙, 최소 검증 기준, 응답 스타일을 반영했다.
- 히스토리 markdown 파일 보관 위치를 `codex_history/`로 일원화했다.

### 보완하면 좋을만한점
- 이후 `src/alpamayo1_5/`, `tools/`, `tests/`별로 더 구체적인 하위 `AGENTS.md`를 추가하면 작업 정확도를 더 높일 수 있다.
- `rules` 파일까지 추가하면 승인 병목을 더 줄일 수 있다.

### 다음스텝
- 필요하면 `src/alpamayo1_5/`와 `tools/`에 폴더 전용 `AGENTS.md`를 추가한다.
- 자주 쓰는 안전 명령에 대한 `rules` 초안을 별도로 만든다.

## 2026-04-15 (Alpamayo TRT-LLM PyTorch backend 통합 진행)

### 시도해본것
- `src/alpamayo1_5/models/trtllm_backend.py`를 추가해 Alpamayo `generate_text()` 경로가 `TensorRT-LLM`의 `pytorch backend`를 직접 호출할 수 있는 lazy adapter를 만들었다.
- `ReasoningVLAConfig`에 `use_trtllm_vlm_pytorch_backend`, `trtllm_vlm_model_dir`, `trtllm_vlm_max_*`, `trtllm_vlm_disable_overlap_scheduler`, `trtllm_vlm_disable_flashinfer_sampling` 설정을 추가했다.
- `src/alpamayo1_5/models/base_model.py`의 `generate_text()`에 TRT-LLM backend 우회 경로를 넣었다.
- `tools/export_alpamayo_expert_to_qwen3.py`를 추가해 Alpamayo `expert.*`를 별도 Qwen3 HF 체크포인트로 재포장하는 스크립트를 만들었다.
- `tools/run_trtllm_pytorch_qwen3_smoke.py`를 추가해 expert/text 계열 체크포인트를 TRT-LLM `pytorch backend`로 태우는 smoke 경로를 만들었다.
- `tools/run_trtllm_pytorch_qwen3vl_smoke.py`와 `tools/run_trtllm_pytorch_qwen3_smoke.py`에 `--disable-flashinfer-sampling` 옵션을 추가했다.
- 로컬 GPU에서 repacked `Qwen3-VL`과 repacked `expert`를 실제로 TRT-LLM `pytorch backend`에 태워 smoke test를 수행했다.
- `flashinfer` 샘플러 문제를 피하기 위해 로컬 TRT-LLM 코드에서 이미 지원하는 `disable_flashinfer_sampling` 옵션을 찾아 adapter와 smoke 스크립트에 반영했다.
- `Qwen3` export config의 `rope_scaling`을 TRT-LLM이 이해하는 `mrope` 형태로 정규화하도록 `tools/export_qwen3vl_text_decoder.py`와 `tools/export_alpamayo_expert_to_qwen3.py`를 수정했다.
- repacked expert 체크포인트를 `/home/jys/workspace/alpamayo_qwen3_expert_hf`로 실제 생성했다.
- `expert`용 `embed_tokens`와 `lm_head`를 hidden size `2048`에 맞는 dummy tensor로 생성하도록 export 스크립트를 수정했다.
- 수정된 expert 체크포인트로 TRT-LLM `pytorch backend` executor와 generation smoke를 다시 수행했다.

### 실패한점
- 초기 `expert` export는 Alpamayo top-level config에 `text_config`가 없어서 실패했다. VLM repacked config의 `text_config`를 기준으로 다시 묶어 해결했다.
- `expert` checkpoint를 TRT-LLM `pytorch backend`에 바로 태우면 `rope_type=default`를 이해하지 못해 executor 초기화가 실패했다.
- `rope_scaling`을 `mrope`로 정규화한 뒤에는 더 깊은 구조 mismatch가 드러났다. `expert`의 hidden size는 `2048`인데 내가 끌어온 `vlm`의 `embed_tokens`와 `lm_head`는 `4096` 차원이라 weight load에서 깨진다.
- 즉 `expert`는 HF `Qwen3ForCausalLM` 포맷처럼 보이게 만드는 것만으로는 부족하고, Alpamayo 원래 구조처럼 `inputs_embeds` 전용 경로를 유지해야 한다.
- `Qwen3-VL` smoke는 기본 sampler에서 `flashinfer`를 타다가 `FlashInfer requires GPUs with sm75 or higher`로 실패했다.
- `expert`를 VLM 쪽 `embed_tokens`/`lm_head`와 함께 묶는 초기 설계는 hidden size mismatch (`2048` vs `4096`) 때문에 실패했다.

### 성공한점
- Alpamayo product code 안에 TRT-LLM `pytorch backend`를 직접 호출하는 VLM adapter를 넣었다.
- 로컬 `Qwen3-VL` repacked checkpoint는 `disable_flashinfer_sampling=True` 조건에서 TRT-LLM `pytorch backend`에서 실제 응답을 반환했다.
- 해당 응답은 2026-04-15 기준 로컬 GPU에서 다음 형태로 끝까지 완료됐다.
  - prompt: `What is visible in this driving scene?`
  - output token ids: `[153165, 153166, 153172, 153172, 153172, 153172, 153172, 153172]`
  - output text: `<i1496><i1497><i1503><i1503><i1503><i1503><i1503><i1503>`
- 즉 `Qwen3-VL backbone`은 현재 환경에서 “정적 engine”이 아니라 “TRT-LLM PyTorch backend + flashinfer sampling 비활성화” 경로로는 실제 구동 가능함을 확인했다.
- `expert` 쪽도 단순한 unsupported 문제가 아니라, 더 구체적으로는 `embed/lm_head` 차원 불일치가 blocker라는 점을 코드/실행 로그로 좁혔다.
- `expert`는 dummy-compatible export로 다시 내리면 TRT-LLM `pytorch backend`에서 실제 executor 초기화와 generation까지 완료된다.
- 2026-04-15 기준 `/home/jys/workspace/alpamayo_qwen3_expert_hf`에 대해 다음 smoke 결과를 얻었다.
  - prompt: `Summarize the scene briefly.`
  - output token ids: `[125726, 71531, 64205, 96097, 20281, 146917, 57082, 5953]`
  - output text: `ỡassociate消息 Tup:^(ဘigator�`
- 이 결과는 품질 검증용이 아니라 “Alpamayo expert transformer를 TRT-LLM Qwen3 runtime에 태울 수 있는 구조적 호환성”을 확인한 것이다.

### 보완하면 좋을만한점
- `generate_text()`가 실제 notebook/VQA 입력에서도 동작하는지 Alpamayo inference 경로로 한 번 더 검증하면 좋다.
- TRT-LLM VLM adapter에 `shutdown()` 호출 지점을 넣거나 inference 종료 시점 정리를 위한 헬퍼를 추가하면 worker 누수를 줄일 수 있다.
- `expert`는 `embed_tokens`와 `lm_head`가 실제로 필요 없는 구조이므로, dummy weight를 넣어 executor만 세우는 실험과 `inputs_embeds` 전용 custom runtime 실험을 분리해서 진행하는 게 좋다.
- VLM smoke 결과가 trajectory discrete token으로만 나온 만큼, sampling/stop 설정이나 prompt 형식을 Alpamayo VQA 메시지에 더 가깝게 맞춰 품질 확인을 더 해야 한다.
- 현재 GPU/driver 조합에서 `flashinfer` sampler가 깨지므로, 이후 작업에서는 `disable_flashinfer_sampling`를 기본값으로 보는 편이 안전하다.
- `expert` smoke는 dummy embedding/head를 사용하므로, diffusion 통합 단계에서는 `generate()` 대신 hidden-state 전용 forward 경로를 별도로 만들어야 한다.

### 다음스텝
- Alpamayo `generate_text()`를 실제 입력 샘플로 호출해 TRT-LLM backend 통합이 제품 코드 수준에서도 동작하는지 확인한다.
- `expert`는 이제 executor까지 올라오므로, 다음 단계는 `LLM.generate()`가 아니라 `inputs_embeds + custom position_ids + custom attention_mask`를 받는 전용 TRT-LLM `_torch` wrapper 방향으로 내려간다.
- 각 단계가 통과할 때마다 git 커밋을 남겨 언제든지 rollback 가능한 기준점을 유지한다.
