import run_Exp # 리팩토링된 run_longExp.py 파일을 import

def dict_to_arglist(args_dict:dict):
    """
    딕셔너리를 '--key value' 형태의 리스트로 변환합니다.
    None, False, 또는 0과 같이 argparse의 기본값을 사용해야 하는 값은 리스트에 추가하지 않습니다.
    """
    arg_list = []
    
    for key, value in args_dict.items():
        # 1. key는 '--' prefix를 붙입니다.
        key_str = f'--{key}'

        # 2. store_true/store_false와 같은 '액션' 인자 처리 (값을 가지지 않음)
        # False인 경우 (즉, 기본값을 유지하려는 경우) 인자 리스트에 추가하지 않습니다.
        if isinstance(value, bool):
            if value is True:
                # store_true 인자는 '--key'만 추가하고 value를 추가하지 않습니다.
                arg_list.append(key_str)
            # False인 경우는 (기본값인 경우가 많으므로) 추가하지 않고 건너뜁니다.
            continue 

        # 3. None 또는 기타 '추가하지 않을 값' 처리
        if value is None:
            continue

        # 4. 일반적인 값을 가진 인자 처리
        arg_list.append(key_str)
        arg_list.append(str(value))
        
    return arg_list


def define_base_arguments():
    """모든 실험에 공통으로 적용되는 기본 인자들을 딕셔너리로 정의합니다."""
    # 딕셔너리의 값은 오버라이드할 값만 설정합니다. 
    # None 또는 False는 arglist로 변환되지 않고, run_longExp.py의 default 값이 사용됩니다.
    base_args = {
        # 기본 설정
        'random_seed': 2021,
        'is_training': 1, # True를 의미 (run_longExp.py에서 type=int)
        'model': 'PatchTST',
        'data': 'custom',
        'des': 'Exp',

        # 데이터 경로 및 설정
        'root_path': './dataset/',
        'data_path': 'weather.csv',
        'features': 'M',
        'seq_len': 336,
        
        # 모델 파라미터 (PatchTST)
        'enc_in': 21,
        'e_layers': 3,
        'n_heads': 16,
        'd_model': 128,
        'd_ff': 256,
        'dropout': 0.2,
        'fc_dropout': 0.2,
        'head_dropout': 0.0,
        'patch_len': 16,
        'stride': 8,

        # 최적화 및 학습
        'train_epochs': 100,
        'patience': 20,
        'itr': 1,
        'batch_size': 128,
        'learning_rate': 0.0001,
        
        'label_len': 0, 
        
        # --- store_true/store_false 인자 예시 ---
        # is_training: 1로 설정되어 위에서 처리됨
        # distil: run_longExp.py에서 default=True이므로, False로 설정하려면 'distil': 0 또는 'distil': False로 명시해야 합니다.
        # use_amp: default=False이므로, 켜고 싶으면 'use_amp': True로 설정합니다.
        # do_predict: default=False
        'do_predict': False, # False는 arglist에 포함되지 않아 default(False) 유지
        'use_amp': False    # False는 arglist에 포함되지 않아 default(False) 유지
    }
    return base_args


def run_all_experiments():
    """정의된 인자를 사용하여 모든 pred_len에 대한 실험을 반복 실행하고 결과를 터미널에 출력합니다."""
    
    base_args = define_base_arguments()
    
    pred_lengths = [96, 192, 336, 720]
    
    for pred_len in pred_lengths:
        # 1. 특정 실험 설정 복사 및 업데이트
        current_args = base_args.copy()
        current_args['pred_len'] = pred_len
        
        # 2. model_id 설정
        model_id_name = "weather"
        seq_len = current_args['seq_len']
        
        current_args['model_id'] = f"{model_id_name}_{seq_len}_{pred_len}"
        
        print("\n" + "="*80)
        print(f"  >>> STARTING EXPERIMENT: model={current_args['model']}, pred_len={pred_len} <<<")
        print("="*80)
        
        try:
            # 3. 딕셔너리를 argparse 리스트로 변환
            arg_list_to_pass = dict_to_arglist(current_args)
            
            # 4. run_longExp.py의 main 함수를 직접 호출하고 리스트 인자를 전달
            run_Exp.main(custom_args=arg_list_to_pass)
            
            print(f"\n--- SUCCESSFULLY FINISHED pred_len={pred_len} ---")
            
        except Exception as e:
            print(f"\n--- AN ERROR OCCURRED during pred_len={pred_len} ---")
            print(f"Error details: {e}")

    print("\n" + "#"*80)
    print("  >>> ALL EXPERIMENTS LOOP FINISHED <<<")
    print("#"*80)


if __name__ == '__main__':
    run_all_experiments()