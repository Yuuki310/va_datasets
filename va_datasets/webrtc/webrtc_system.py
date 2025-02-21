import collections
import contextlib
import sys
import wave

import webrtcvad

import pandas as pd

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        # assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n
    
    # yield Frame(audio[offset:offset + n], timestamp, duration)

def vad_collector(sample_rate: int, frame_duration_ms: int,
    padding_duration_ms: int, vad: webrtcvad.Vad, frames: list[Frame],
    voice_trigger_on_thres: float=0.9, voice_trigger_off_thres: float=0.1) -> list[dict]:
    """音声非音声セグメント処理

    Args:
        sample_rate (int): 単位時間あたりのサンプル数[Hz]
        frame_duration_ms (int): フレーム長
        padding_duration_ms (int): ガード長
        vad (webrtcvad.Vad): _description_
        frames (list[Frame]): フレーム分割された音声データ
        voice_trigger_on_thres (float, optional): 音声セグメント開始と判断する閾値. Defaults to 0.9.
        voice_trigger_off_thres (float, optional): 音声セグメント終了と判断する閾値. Defaults to 0.1.

    Returns:
        list[dict]: セグメント結果
    """
    # ガードするフレーム数
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)

    # バッファ(リングバッファではなくする)
    # ring_buffer = collections.deque(maxlen=num_padding_frames)
    frame_buffer = []

    # いま音声かどうかのトリガのステータス
    triggered = False

    voiced_frames = []
    vu_segments = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        frame_buffer.append((frame, is_speech))

        # 非音声セグメントの場合
        if not triggered:

            # 過去フレームのうち音声判定数を取得
            # 過去を見る数はnum_padding_frames個
            num_voiced = len([f for f, speech in frame_buffer[-num_padding_frames:] if speech])

            # 9割以上が音声の場合は音声にトリガする(立ち上がり)
            if num_voiced > voice_trigger_on_thres * num_padding_frames:
                triggered = True

                # num_padding_framesより前は非音声セグメントとする
                audio_data = b''.join([f.bytes for f, _ in frame_buffer[:-num_padding_frames]])
                vu_segments.append({"vad": 0, "audio_size": len(audio_data), "audio_data": audio_data})

                # num_padding_frames以降は音声セグメント終了時にまとめるため一旦保持
                for f, _ in frame_buffer[-num_padding_frames:]:
                    voiced_frames.append(f)
                frame_buffer = []

        # 音声セグメントの場合
        else:
            # フレームを保持
            voiced_frames.append(frame)

            # 過去フレームのうち非音声判定数を取得
            # 過去を見る数はnum_padding_frames個
            num_unvoiced = len([f for f, speech in frame_buffer[-num_padding_frames:] if not speech])

            # 9割以上が非音声の場合はトリガを落とす(立ち下がり)
            if num_unvoiced > (1 - voice_trigger_off_thres) * num_padding_frames:
                triggered = False

                # 音声セグメントをまとめる
                # audio_data = b''.join([f.bytes for f in voiced_frames])
                audio_data = b''.join([f.bytes for f in voiced_frames[:-num_padding_frames]])
                vu_segments.append({"vad": 1, "audio_size": len(audio_data), "audio_data": audio_data})

                # num_padding_frames以降は非音声セグメント終了時にまとめるため保持

                frame_buffer = []
                for f in voiced_frames[-num_padding_frames:]:
                    frame_buffer.append((f, False))
                voiced_frames = []

                # frame_buffer = []


    # 終了時に音声セグメントか非音声セグメントかどうかで処理を分ける
    if triggered:
        audio_data = b''.join([f.bytes for f in voiced_frames])
        vu_segments.append({"vad": 1, "audio_size": len(audio_data), "audio_data": audio_data})
    else:
        audio_data = b''.join([f.bytes for f, _ in frame_buffer])
        vu_segments.append({"vad": 0, "audio_size": len(audio_data), "audio_data": audio_data})

    return vu_segments


    
def pipeline(
    audio_path, segments_dir, 
    aggresive : int = 3, 
    frame_duration_ms : int = 10
    ):
    import os
    audio, sample_rate = read_wave(audio_path)
    vad = webrtcvad.Vad(aggresive)
    
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    frames = list(frames)
    
    # セグメント結果
    vu_segments = vad_collector(sample_rate, frame_duration_ms, 50, vad, frames)
    # wavファイル格納先作成
    for_df = []
    for i, segment in enumerate(vu_segments):
        path = os.path.join(segments_dir, f"segment-{i:03d}-vad{segment['vad']}.wav")
        write_wave(str(path), segment['audio_data'], sample_rate)
        # print(segment["audio_size"]/2.000/sample_rate)
        for_df.append({
            "filename": os.path.basename(path),
            "vad": segment["vad"],
            "duration_sec": segment["audio_size"]/2.0/sample_rate,
        })
    return for_df
    # df = pl.DataFrame(for_df)
    # print(df.filter(pl.col("vad")==1).sort(pl.col("duration_sec"), descending=True)[-10:])
    # print(df.sort(pl.col("duration_sec"), descending=True)[:10])
    

if __name__ == '__main__':
    main(sys.argv[1:])