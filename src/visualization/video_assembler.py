import moviepy.editor as mp
from pathlib import Path
import os

class VideoAssembler:
    def __init__(self, video_folder="../video_output"):
        self.video_folder = Path(video_folder)
    
    def create_final_video(self):
        """Assemble all components into final educational video"""
        print("Assembling final educational video...")
        
        video_files = []
        
        # Check for generated video files
        intro_file = self.video_folder / "problem_introduction.mp4"
        evolution_file = None
        comparison_file = self.video_folder / "algorithm_comparison.mp4"
        
        # Find the evolution file (may have different naming)
        for file in self.video_folder.glob("ga_evolution_*.mp4"):
            evolution_file = file
            break
        
        clips = []
        
        try:
            # Add intro if exists
            if intro_file.exists():
                print(f"Adding intro: {intro_file}")
                intro_clip = mp.VideoFileClip(str(intro_file))
                clips.append(intro_clip)
            
            # Add evolution if exists
            if evolution_file and evolution_file.exists():
                print(f"Adding evolution: {evolution_file}")
                evolution_clip = mp.VideoFileClip(str(evolution_file))
                clips.append(evolution_clip)
            
            # Add comparison if exists
            if comparison_file.exists():
                print(f"Adding comparison: {comparison_file}")
                comparison_clip = mp.VideoFileClip(str(comparison_file))
                clips.append(comparison_clip)
            
            if not clips:
                print("❌ No video files found to assemble!")
                return None
            
            # Create title screen
            title_screen = self.create_title_screen()
            conclusion_screen = self.create_conclusion_screen()
            
            # Combine all clips
            all_clips = [title_screen] + clips + [conclusion_screen]
            final_video = mp.concatenate_videoclips(all_clips)
            
            # Export final video
            output_path = self.video_folder / "GA_2TP_Educational_Video.mp4"
            print(f"Exporting final video to: {output_path}")
            
            final_video.write_videofile(
                str(output_path),
                fps=24,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Clean up
            for clip in all_clips:
                clip.close()
            
            print(f"✅ Final educational video created: {output_path}")
            return final_video
            
        except Exception as e:
            print(f"❌ Error creating final video: {e}")
            return None
    
    def create_title_screen(self, duration=3):
        """Create opening title screen"""
        try:
            # Create a simple color clip as title screen
            title_clip = mp.ColorClip(
                size=(1920, 1080), 
                color=(26, 26, 26), 
                duration=duration
            )
            
            # Add text if possible
            try:
                title_text = mp.TextClip(
                    "Genetic Algorithm for 2-Transfer Problem\nEducational Demonstration",
                    fontsize=50,
                    color='white',
                    font='Arial-Bold'
                ).set_duration(duration).set_position('center')
                
                title_screen = mp.CompositeVideoClip([title_clip, title_text])
            except:
                # Fallback to just color clip if text fails
                title_screen = title_clip
            
            return title_screen
            
        except Exception as e:
            print(f"Warning: Could not create title screen: {e}")
            return mp.ColorClip(size=(640, 480), color=(0, 0, 0), duration=1)
    
    def create_conclusion_screen(self, duration=3):
        """Create conclusion screen"""
        try:
            conclusion_clip = mp.ColorClip(
                size=(1920, 1080), 
                color=(26, 26, 26), 
                duration=duration
            )
            
            try:
                conclusion_text = mp.TextClip(
                    "Thank you for watching!\nGA-2TP Research Demonstration",
                    fontsize=40,
                    color='white',
                    font='Arial'
                ).set_duration(duration).set_position('center')
                
                conclusion_screen = mp.CompositeVideoClip([conclusion_clip, conclusion_text])
            except:
                conclusion_screen = conclusion_clip
            
            return conclusion_screen
            
        except Exception as e:
            print(f"Warning: Could not create conclusion screen: {e}")
            return mp.ColorClip(size=(640, 480), color=(0, 0, 0), duration=1)

def main():
    assembler = VideoAssembler()
    assembler.create_final_video()

if __name__ == "__main__":
    main()
