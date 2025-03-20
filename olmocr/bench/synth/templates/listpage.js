//import React from 'react';

const PermitGuidelinesTemplate = () => {
  // Sample data - you can replace these with your own
  const guidelineItems = [
    {
      number: 'iii.',
      content: 'Not rely on personal preference or opinion, or regional interpretation of statute, regulation or guidance that is inconsistent with the Department\'s statewide interpretation. Staff should confer with the appropriate Bureau Director as necessary.'
    },
    {
      number: 'iv.',
      content: 'Process technically adequate and scientifically sound applications for final approval to minimize elapsed time in accordance with the Permit Decision Guarantee.'
    },
    {
      number: 'v.',
      content: 'Where the Application Manager determines that the technical information submitted with the application does not meet technical guidance or standards published by the Department, the application must provide the scientific or engineering basis to support the application. Note that deviations from technical guidance can generally be approved, by the appropriate section chief and manager, when warranted, provided acceptable justification has been submitted. Minor deficiencies that can be easily corrected should be addressed through a telephone call with the applicant and consultant, and may negate the need for a deficiency letter. The Program Manager or District Manager will be responsible for making that decision.'
    },
    {
      number: 'vi.',
      content: 'If an application fails to provide the technical information necessary to document that applicable regulatory and statutory requirements will be achieved, it is technically deficient and the Application Manager will prepare a technical deficiency letter. Again, all deficiencies noted must cite the statutory or regulatory obligation that the application has failed to meet and the Section Chief and the Program Manager will routinely review these letters. For District Oil and Gas Offices and District Mining Offices the Permits Chief and the Manager will review the letters.'
    },
    {
      number: 'vii.',
      content: 'Applicant responses that do not make the application technically adequate within the established response timeframe will be subject to the Elevated Review Process below. Applications that are made technically adequate within the established response timeframe will proceed to processing for final action.'
    }
  ];

  // Footnote data
  const footnote = {
    number: '2',
    content: 'More technically complex projects and applications may receive additional deficiency letters as appropriate prior to a decision point. This exception will not void inclusion in the Permit Decision Guarantee and will follow program specific guidance that is developed. The more technically complex projects and applications are noted with an asterisk ("*") in Appendix A.'
  };

  // Document info
  const documentInfo = "021-2100-001 / November 2, 2012 / Page 11";

  // Special note about technical deficiency letter
  const technicalDeficiencyNote = {
    prefix: 'One',
    superscript: '2',
    content: ' technical deficiency letter will be sent. Each deficiency cited must note the statute, regulation or technical guidance provision. Technical guidance provides a means to compliance, but may not be used or cited when issuing a permit denial. The letter will state, as necessary, that the Permit Decision Guarantee is no longer applicable and offer the applicant an opportunity to meet and discuss the deficiencies. The letter will include a deadline for submission of the deficient information.'
  };

  return (
    <div className="bg-white p-8 max-w-4xl mx-auto font-serif text-black">
      <div className="mb-8">
        {guidelineItems.map((item, index) => (
          <div key={index} className="mb-6 flex">
            <div className="w-12 flex-shrink-0 font-bold">{item.number}</div>
            <div className="flex-grow">{item.content}</div>
          </div>
        ))}
        
        {/* Technical deficiency letter note */}
        <div className="mb-6 ml-12">
          <p>
            {technicalDeficiencyNote.prefix}
            <sup>{technicalDeficiencyNote.superscript}</sup>
            {technicalDeficiencyNote.content}
          </p>
        </div>
      </div>
      
      {/* Horizontal line */}
      <div className="border-t border-gray-400 my-6"></div>
      
      {/* Footnote section */}
      <div className="text-sm">
        <p>
          <sup>{footnote.number}</sup> {footnote.content}
        </p>
      </div>
      
      {/* Document info */}
      <div className="text-center mt-6 text-sm">
        {documentInfo}
      </div>
    </div>
  );
};

//export default PermitGuidelinesTemplate;
window.BookPageTemplate = PermitGuidelinesTemplate;