import { useEffect, useRef } from 'react';

export default function HeroVideo() {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = 4.0;
    }
  }, []);

  return (
    <>
      <div className="hero-video-background not-content"></div>
      <div className="hero-terminal-container not-content">
        <div className="terminal-header">
          <div className="terminal-dot red"></div>
          <div className="terminal-dot yellow"></div>
          <div className="terminal-dot green"></div>
        </div>
        <div className="hero-video-wrapper">
          <video
            ref={videoRef}
            autoPlay
            muted
            loop
            playsInline
            className="hero-video"
          >
            <source src="/sparkrun-terminal.mp4" type="video/mp4" />
          </video>
        </div>
      </div>
    </>
  );
}
