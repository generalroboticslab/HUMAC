using System;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Collections.Generic;

namespace Examples.Bowling
{
    public enum ScoreType
    {
        Open = 0,
        Spare = 1,
        Strike = 2,
    }

    public class Score
    {
        public static readonly int MaxScore = 10;
        public static readonly int MaxFrames = 10;

        public int TotalScore { get; private set; } = 0;

        public int CurrentFrame { get; private set; } = 0;
        public int CurrentRoll { get; private set; } = 0;

        private int _prevClearedPins = 0;
        private readonly List<Tuple<ScoreType, int>> _scoreByShots = new();

        public void TakeNewRoll(int clearedPins)
        {
            var score = clearedPins - _prevClearedPins;
            if (clearedPins >= MaxScore)
            {
                _prevClearedPins = 0;
            }
            else
            {
                _prevClearedPins = clearedPins;
            }
            // update current state
            if (score == MaxScore)
            {
                _scoreByShots.Add(Tuple.Create(ScoreType.Strike, score));
                CurrentRoll = 2;
            }
            else
            {
                if (CurrentRoll == 1 && _scoreByShots.Last().Item2 + score == MaxScore)
                {
                    _scoreByShots.Add(Tuple.Create(ScoreType.Spare, score));
                }
                else
                {
                    _scoreByShots.Add(Tuple.Create(ScoreType.Open, score));
                }
                CurrentRoll++;
            }
            // update previous rolls
            var count = _scoreByShots.Count;
            if (count > 1 && _scoreByShots[count - 2].Item1 != ScoreType.Open && CurrentRoll <= 1)
            {
                _scoreByShots[count - 2] = Tuple.Create(_scoreByShots[count - 2].Item1, _scoreByShots[count - 2].Item2 + score);
            }
            if (count > 2 && _scoreByShots[count - 3].Item1 == ScoreType.Strike)
            {
                _scoreByShots[count - 3] = Tuple.Create(_scoreByShots[count - 3].Item1, _scoreByShots[count - 3].Item2 + score);
            }
            // update total score
            TotalScore = _scoreByShots.Sum(p => p.Item2);
        }

        public bool CheckFrameOver()
        {
            if (CurrentFrame == MaxFrames - 1)
            {
                if (CurrentRoll > 2)
                {
                    CurrentRoll = 0;
                    _prevClearedPins = 0;
                    CurrentFrame++;
                    return true;
                }
                else if (CurrentRoll > 1 && _scoreByShots.Last().Item1 == ScoreType.Open)
                {
                    CurrentRoll = 0;
                    _prevClearedPins = 0;
                    CurrentFrame++;
                    return true;
                }
            }
            else
            {
                if (CurrentRoll > 1)
                {
                    CurrentRoll = 0;
                    _prevClearedPins = 0;
                    CurrentFrame++;
                    return true;
                }
            }

            return false;
        }

        public bool CheckEpisodeOver()
        {
            return CurrentFrame >= MaxFrames;
        }

        public void Reset()
        {
            TotalScore = 0;
            CurrentFrame = 0;
            CurrentRoll = 0;
            _prevClearedPins = 0;
            _scoreByShots.Clear();
        }

        public void Encode(BinaryWriter writer)
        {
            writer.Write(TotalScore);
            writer.Write((byte)CurrentFrame);
        }

        public void Decode(BinaryReader reader)
        {
            TotalScore = reader.ReadInt32();
            CurrentFrame = reader.ReadByte();
        }

        //public void Test()
        //{
        //    bool over;
        //    for (var i = 0; i < 9; ++i)
        //    {
        //        TakeNewRoll(9);
        //        TakeNewRoll(1);
        //        over = CheckFrameOver();
        //        Debug.Assert(over);
        //        over = CheckEpisodeOver();
        //        Debug.Assert(!over);
        //    }
        //    TakeNewRoll(9);
        //    TakeNewRoll(1);
        //    over = CheckFrameOver();
        //    Debug.Assert(!over);
        //    TakeNewRoll(9);
        //    over = CheckFrameOver();
        //    Debug.Assert(over);
        //    over = CheckEpisodeOver();
        //    Debug.Assert(over);
        //    Reset();
        //}
    }
}