from pydub import AudioSegment
import librosa
import soundfile as sf

if __name__ == "__main__":
    
    paths = [
        "/data/group1/z40351r/datasets_turntaking/data/CEJC2021/data/T020/T020_022/T020_022a_IC01.wav",
        "/data/group1/z40351r/datasets_turntaking/data/CEJC2021/data/T020/T020_022/T020_022a_IC02.wav",
        "/data/group1/z40351r/datasets_turntaking/data/CEJC2021/data/T020/T020_022/T020_022c_IC01.wav",
        "/data/group1/z40351r/datasets_turntaking/data/CEJC2021/data/T020/T020_022/T020_022c_IC02.wav",        
    ]
    for sound_path in paths:
        # ステレオ音源ファイルを読み込む
        stereo_audio = AudioSegment.from_file(sound_path, format="wav")

        # ステレオ音源をモノラルに変換
        mono_audio = stereo_audio.set_channels(1)

        # モノラル音源を一時的にWAVファイルとして保存
        mono_audio.export("mono_audio.wav", format="wav")

        # モノラル音源をダウンサンプリング（例: 16 kHzに設定）
        audio, sample_rate = librosa.load("mono_audio.wav", sr=16000)

        # ダウンサンプリング後の音源を保存
        sf.write(sound_path, audio, sample_rate)