//import React from 'react';

const BookPageTemplate = () => {
  // Only three state variables as requested
  const [title, setTitle] = React.useState("ADVENTURES OF DON QUIXOTE");
  const [pageNumber, setPageNumber] = React.useState("289");
  const [text, setText] = React.useState(
    "deed,\" said Don Quixote, \"thou hast hit the point, Sancho, which can alone shake my resolution; I neither can, nor ought to, draw my sword, as I have often told thee, against those who are not dubbed knights. To thee which I had premeditated, thy share of the booty would have been at least the emperor's crown of gold and Cupid's painted wings; for I would have plucked them off perforce, and delivered them into thy hands.\" \"The"
  );

  // Styles for heavily degraded scan effect
  const heavilyDegradedStyles = {
    filter: 'grayscale(30%) contrast(120%) brightness(85%) sepia(20%)',
    position: 'relative',
    backgroundColor: '#e6ddc6', // More yellowed aged paper
    backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=\'0 0 200 200\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cfilter id=\'noiseFilter\'%3E%3CfeTurbulence type=\'fractalNoise\' baseFrequency=\'0.85\' numOctaves=\'3\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23noiseFilter)\' opacity=\'0.25\'/%3E%3C/svg%3E")',
    boxShadow: 'inset 0 0 70px rgba(0, 0, 0, 0.3), 0 0 5px rgba(0,0,0,0.1)',
    padding: '32px',
    borderRadius: '2px',
    overflow: 'hidden',
    transform: 'rotate(0.3deg)', // Slightly askew scan
  };

  // Heavily degraded text
  const badScanTextStyle = {
    fontFamily: '"Times New Roman", serif',
    letterSpacing: '-0.01em',
    wordSpacing: '0.02em',
    fontWeight: '500',
    color: '#222222',
    textShadow: '0 0 1px rgba(0, 0, 0, 0.5)',
    transform: 'scale(1.01, 0.99) rotate(-0.4deg)', // Distorted proportions
  };

  // Random coffee stain effect
  const coffeeStain = {
    position: 'absolute',
    width: '100px',
    height: '80px',
    top: '25%',
    right: '15%',
    borderRadius: '50%',
    background: 'radial-gradient(ellipse at center, rgba(139,69,19,0.15) 0%, rgba(139,69,19,0.1) 50%, rgba(139,69,19,0.05) 70%, rgba(139,69,19,0) 100%)',
    transform: 'rotate(30deg) scale(1.5, 1)',
    pointerEvents: 'none',
    zIndex: 1,
  };

  // Water damage effect
  const waterDamage = {
    position: 'absolute',
    width: '70%',
    height: '40%',
    bottom: '10%',
    left: '5%',
    opacity: 0.07,
    background: 'radial-gradient(ellipse at center, rgba(0,0,0,0.2) 0%, rgba(0,0,0,0.1) 40%, rgba(0,0,0,0) 70%)',
    borderRadius: '40% 60% 70% 30% / 40% 50% 60% 50%',
    pointerEvents: 'none',
    zIndex: 1,
  };

  // Add fold lines
  const foldLine = {
    position: 'absolute',
    width: '100%',
    height: '3px',
    top: '30%',
    left: 0,
    background: 'linear-gradient(to right, rgba(0,0,0,0) 0%, rgba(0,0,0,0.03) 20%, rgba(0,0,0,0.08) 50%, rgba(0,0,0,0.03) 80%, rgba(0,0,0,0) 100%)',
    boxShadow: '0 1px 3px rgba(255,255,255,0.2)',
    pointerEvents: 'none',
    zIndex: 2,
  };

  // Torn edge effect
  const tornEdge = {
    position: 'absolute',
    top: 0,
    right: 0,
    width: '100%',
    height: '100%',
    background: 'linear-gradient(135deg, transparent 97%, #e6ddc6 97%, #e6ddc6 100%)',
    pointerEvents: 'none',
  };

  return (
    <div style={{
      maxWidth: '800px',
      margin: '0 auto',
      padding: '16px',
    }}>
      {/* Heavily degraded scan container */}
      <div style={heavilyDegradedStyles}>
        {/* Noise overlay */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'url("data:image/png;base64,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") repeat',
          opacity: 0.15,
          pointerEvents: 'none',
        }}></div>
        
        {/* Scan lines effect */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'linear-gradient(to bottom, rgba(0,0,0,0.03) 1px, transparent 1px)',
          backgroundSize: '100% 2px',
          opacity: 0.5,
          pointerEvents: 'none',
        }}></div>
        
        {/* Add coffee stain */}
        <div style={coffeeStain}></div>
        
        {/* Add water damage */}
        <div style={waterDamage}></div>
        
        {/* Add fold line */}
        <div style={foldLine}></div>
        
        {/* Add torn edge */}
        <div style={tornEdge}></div>
        
        {/* Header with skewed alignment */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '2px solid #000',
          paddingBottom: '4px',
          marginBottom: '24px',
          position: 'relative',
          opacity: 0.8,
          transform: 'skew(-0.5deg, 0.3deg)',
        }}>
          <div style={{width: '48px'}}></div>
          <h1 style={{
            ...badScanTextStyle,
            fontSize: '20px',
            fontWeight: 'bold',
            textAlign: 'center',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            opacity: 0.8,
          }}>{title}</h1>
          <div style={{
            ...badScanTextStyle,
            fontSize: '20px', 
            fontWeight: 'bold',
            opacity: 0.85,
          }}>{pageNumber}</div>
        </div>
        
        {/* Horizontal divider with uneven quality */}
        <div style={{
          borderBottom: '1px solid #444',
          marginBottom: '24px',
          opacity: 0.6,
          filter: 'blur(0.3px)',
          transform: 'scaleY(1.5) skew(0.7deg)',
        }}></div>
        
        {/* Text content with severely degraded appearance */}
        <div style={{
          columnCount: 2,
          columnGap: '20px',
          columnRule: '1px solid rgba(0,0,0,0.1)',
          textAlign: 'justify',
          ...badScanTextStyle,
          fontSize: '16px',
          lineHeight: '1.5',
          opacity: 0.78,
          // Very uneven ink distribution with blurry and faded parts
          WebkitMaskImage: 'linear-gradient(to bottom, rgba(0,0,0,0.9), rgba(0,0,0,0.75) 50%, rgba(0,0,0,0.85))',
          // Text distortion
          filter: 'blur(0.2px)',
        }}>
          {/* Bad scan text with random character fading */}
          <p>{text.split('').map((char, index) => {
            const opacity = Math.random() > 0.8 ? 0.4 + Math.random() * 0.5 : 0.9 + Math.random() * 0.1;
            const blur = Math.random() > 0.95 ? 1 : 0;
            return <span key={index} style={{opacity, filter: `blur(${blur}px)`}}>{char}</span>;
          })}</p>
        </div>
        
        {/* Extra random ink spill */}
        <div style={{
          position: 'absolute',
          width: '10px',
          height: '20px',
          top: '60%',
          left: '25%',
          background: 'rgba(0,0,0,0.3)',
          borderRadius: '50%',
          transform: 'rotate(30deg)',
          filter: 'blur(1px)',
          zIndex: 3,
        }}></div>
      </div>
      
    </div>
  );
};

//export default BookPageTemplate;
window.BookPageTemplate = BookPageTemplate;